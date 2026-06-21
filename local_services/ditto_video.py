"""Ditto local lip-sync avatar as a Pipecat FrameProcessor.

Sits between TTS and transport.output(). Streams the TTS audio to the local Ditto
server (local_services/ditto_server/) over a websocket, receives lip-synced RGB
video frames back, and pushes both the video (OutputImageRawFrame) and the audio
(TTSAudioRawFrame) downstream so the browser plays them in sync.

A/V sync is **frame-clocked**, not time-guessed. The render is bursty and
edge-of-realtime, so forwarding the voice on a fixed delay always drifts. Instead:
- The voice copy bound for the server (for lip-sync) is sent *immediately*.
- The voice copy bound downstream (for the browser) is **buffered** and released
  paced to the *real frames the server reports* having rendered, via its
  video_start / video_clock{frames} / video_end markers: audio for second S is
  released once the server has rendered >= S*fps frames (plus a small lip lead).
  So the voice waits when the render stalls and catches up when it resumes.
- Video frames stream from the server at a fixed FPS and are emitted as they
  arrive (the server paces them; they are not buffered here).
- On interruption (barge-in) we reset the server session and drop buffered voice.
- Fallback: if the server sends no markers within DITTO_SYNC_FALLBACK_S, the voice
  is forwarded unsynced so it never goes silent.

Requires: `pip install websockets`.
"""
from __future__ import annotations

import asyncio
import json
import os

import numpy as np
import websockets
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    OutputImageRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

DITTO_SR = 16000  # hubert (server-side) expects 16 kHz mono


def _to_16k_mono_pcm(audio: bytes, in_rate: int, channels: int) -> bytes:
    """Resample int16 PCM to 16 kHz mono for the Ditto server. TTS often emits
    24 kHz; per-chunk linear resampling is sufficient for the audio encoder."""
    a = np.frombuffer(audio, dtype=np.int16)
    if a.size == 0:
        return b""
    if channels and channels > 1:
        a = a.reshape(-1, channels).mean(axis=1)
    if in_rate and in_rate != DITTO_SR:
        n_out = int(round(a.shape[0] * DITTO_SR / in_rate))
        if n_out <= 0:
            return b""
        src = np.arange(a.shape[0], dtype=np.float64)
        dst = np.linspace(0, a.shape[0] - 1, num=n_out)
        a = np.interp(dst, src, a)
    return a.astype(np.int16).tobytes()


class DittoVideoService(FrameProcessor):
    def __init__(
        self,
        *,
        base_url: str,
        fps: float | None = None,            # OUTPUT fps; server frame-drops 25 -> this
        image_size: tuple[int, int] = (512, 512),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._ws_url = base_url.replace("http", "ws", 1).rstrip("/") + "/stream"
        # ONE fps source of truth. The release clock (video_frames/_fps), the server's
        # frame-drop stride (round(25/fps)), and the transport's video_out_framerate MUST
        # all be the same number or audio/video drift. The factory (stages/avatar.py) passes
        # DITTO_FPS; fall back to the same env here so there's no stale hard-coded divergence.
        if fps is None:
            fps = float(os.getenv("DITTO_FPS", "12"))
        self._fps = fps
        self._size = image_size
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._recv_task: asyncio.Task | None = None
        self._closing = False
        # Frame-clocked A/V sync. The voice is buffered and released paced to the
        # real video the server reports rendering. DITTO_SYNC_LEAD_S is a small
        # constant lip lead (raise/lower so lips align). DITTO_SYNC_FALLBACK_S is
        # the grace period before giving up on markers and forwarding unsynced.
        # overlap=25 (the server default) has a measured intrinsic lip offset of ~0.00s,
        # so no lead compensation is needed (avatar_tune `align`). Raise/lower only if the
        # browser shows a constant lip lead/trail.
        self._lead_s = float(os.getenv("DITTO_SYNC_LEAD_S", "0.0"))
        # In compute-first mode the server legitimately delays video_start by a
        # sentence's render time, so the no-markers fallback must be patient (10s
        # covers one sentence's render; still a real safety net if the server dies).
        # Compute-first delays video_start by a sentence's render time; a long sentence can
        # take >10s, so the no-markers fallback must be patient or it wrongly gives up and
        # plays the voice UNSYNCED for the rest of that turn (the 11s-reply desync bug). 25s
        # covers a long sentence's pre-render; still a real safety net if the server dies.
        self._fallback_s = float(os.getenv("DITTO_SYNC_FALLBACK_S", "25.0"))
        # Per-turn voice buffer: list of (cumulative_end_s, frame, direction),
        # ordered. cumulative_end_s = audio seconds elapsed at the END of a chunk.
        self._abuf: list[tuple[float, Frame, FrameDirection]] = []
        self._audio_clock_s = 0.0       # running audio seconds buffered this turn
        self._video_frames = 0          # real frames the server reports this turn
        self._turn_active = False       # between TTSStarted and buffer-drained
        self._unsynced = False          # fallback engaged for this turn
        self._first_buffered_t: float | None = None  # when first chunk was buffered
        # A/V LOCK (sync_with_audio): the robust "always match" mode. Instead of pushing
        # video on its own transport clock (which drifts vs audio), we buffer each turn's
        # real frames and release them tagged sync_with_audio=True right AFTER the matching
        # voice -- so pipecat shows each frame at its exact audio position (transport audio
        # queue, not an independent video clock). Fixes lips-lag-voice drift in the browser.
        # Does NOT remove the per-turn render warmup (that's the GPU floor). Flag-gated so we
        # can A/B vs the old immediate-push path instantly.
        self._sync_av = (os.getenv("DITTO_SYNC_WITH_AUDIO", "1") or "1").lower() in ("1", "true", "yes", "on")
        self._vbuf: list[bytes] = []    # per-turn pending video frames (sync_av mode)
        self._video_active = False      # between video_start and video_end markers
        self._fallback_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()     # guards the buffer (recv vs process)
        self._dbg_audio = 0  # diagnostic: count TTS audio frames seen
        self._last_hold_log = 0.0  # throttle for the voice-vs-video hold trace

    # --- connection lifecycle ---------------------------------------------
    async def _connect(self):
        if self._recv_task is not None:
            return
        self._closing = False
        try:
            await self._open_ws()
        except Exception as e:  # noqa: BLE001 — loop will retry/backoff
            logger.warning(f"Ditto initial connect failed ({e!r}); loop will retry.")
        self._recv_task = asyncio.create_task(self._receive_loop())

    async def _open_ws(self):
        logger.info(f"Connecting to Ditto server at {self._ws_url}")
        self._ws = await websockets.connect(
            self._ws_url, max_size=None, ping_interval=None, close_timeout=1
        )
        await self._ws.send(json.dumps({"type": "config", "fps": self._fps}))

    async def _disconnect(self):
        self._closing = True
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._recv_task = None
        if self._ws:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001
                pass
            self._ws = None

    async def _receive_loop(self):
        """Read from the server and push downstream, reconnecting if the socket
        drops so the avatar keeps moving across the whole session.

        Binary messages are RGB video frames (emitted immediately). Text messages
        are sync markers (video_start/video_clock/video_end) that clock the voice
        release -- see _handle_marker."""
        while not self._closing:
            try:
                if self._ws is None:
                    await self._open_ws()
                assert self._ws is not None
                async for message in self._ws:
                    if isinstance(message, bytes):
                        if self._sync_av and self._video_active:
                            # In a turn: buffer; released with the voice on the next
                            # video_clock marker (tagged sync_with_audio) so it displays
                            # locked to its audio moment instead of on the video clock.
                            self._vbuf.append(message)
                        else:
                            # Idle frames (between turns) or legacy mode: show immediately
                            # so the breathing/idle face still animates and the track stays live.
                            await self.push_frame(
                                OutputImageRawFrame(
                                    image=message, size=self._size, format="RGB"
                                ),
                                FrameDirection.DOWNSTREAM,
                            )
                    else:
                        await self._handle_marker(message)
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001
                if not self._closing:
                    logger.warning(f"Ditto ws dropped ({e!r}); reconnecting…")
            finally:
                ws, self._ws = self._ws, None
                if ws is not None:
                    try:
                        await ws.close()
                    except Exception:  # noqa: BLE001
                        pass
            if not self._closing:
                await asyncio.sleep(0.5)

    async def _reset_session(self):
        if self._ws:
            try:
                await self._ws.send(json.dumps({"type": "reset"}))
            except Exception:  # noqa: BLE001
                pass

    # --- frame-clocked voice buffer (A/V sync) ----------------------------
    # The voice is buffered per turn and released paced to the real video the
    # server reports. allowed_s = video_frames / fps + lead; any buffered chunk
    # whose cumulative_end_s <= allowed_s is released, in order.
    def _reset_turn(self):
        self._abuf.clear()
        self._vbuf.clear()
        self._video_active = False
        self._audio_clock_s = 0.0
        self._video_frames = 0
        self._turn_active = False
        self._unsynced = False
        self._first_buffered_t = None

    def _arm_fallback(self):
        """Start the no-markers fallback timer on the first buffered item."""
        if self._first_buffered_t is None:
            self._first_buffered_t = asyncio.get_running_loop().time()
            self._fallback_task = asyncio.create_task(self._fallback_watch())

    async def _buffer_audio(self, frame: TTSAudioRawFrame, direction: FrameDirection):
        """Append a voice chunk to the per-turn buffer, advancing the audio clock
        by the chunk's duration. If the fallback already engaged, forward now."""
        if self._unsynced:
            await self.push_frame(frame, direction)
            return
        sr = getattr(frame, "sample_rate", DITTO_SR) or DITTO_SR
        ch = getattr(frame, "num_channels", 1) or 1
        n = len(frame.audio) // (2 * ch)  # int16 samples per channel
        self._audio_clock_s += n / sr
        async with self._lock:
            self._abuf.append((self._audio_clock_s, frame, direction))
            self._arm_fallback()

    async def _buffer_marker(self, frame: Frame, direction: FrameDirection):
        """Buffer a TTSStarted/Stopped marker at the current audio clock so it
        releases in order with the voice (or forward now if already unsynced)."""
        if self._unsynced:
            await self.push_frame(frame, direction)
            return
        async with self._lock:
            self._abuf.append((self._audio_clock_s, frame, direction))
            self._arm_fallback()

    async def _release_up_to(self, allowed_s: float):
        """Release (push downstream) every buffered chunk due by allowed_s."""
        async with self._lock:
            i = 0
            while i < len(self._abuf) and self._abuf[i][0] <= allowed_s:
                i += 1
            due, self._abuf = self._abuf[:i], self._abuf[i:]
        for _end_s, frame, direction in due:
            await self.push_frame(frame, direction)

    async def _drain_all(self):
        """Flush everything still buffered (turn end / fallback)."""
        async with self._lock:
            due, self._abuf = self._abuf, []
        for _end_s, frame, direction in due:
            await self.push_frame(frame, direction)

    async def _flush_video(self):
        """Push the buffered real frames tagged sync_with_audio so the transport
        displays each at its audio position (locked to the voice, not a video clock).
        Called right AFTER the matching audio is released, so order is audio-then-video."""
        if not self._vbuf:
            return
        async with self._lock:
            pending, self._vbuf = self._vbuf, []
        for img in pending:
            frame = OutputImageRawFrame(image=img, size=self._size, format="RGB")
            frame.sync_with_audio = True
            await self.push_frame(frame, FrameDirection.DOWNSTREAM)

    async def _handle_marker(self, message):
        """React to a server sync marker on the recv loop."""
        try:
            evt = json.loads(message)
        except Exception:  # noqa: BLE001
            return
        kind = evt.get("type")
        if kind == "video_start":
            self._video_frames = 0
            self._video_active = True
            self._vbuf.clear()
        elif kind == "video_clock":
            self._video_frames = int(evt.get("frames", self._video_frames))
            # Order matters for sync_with_audio: release the matching VOICE first, then
            # push the buffered frames -> each frame displays after its audio = locked.
            await self._release_up_to(self._video_frames / self._fps + self._lead_s)
            if self._sync_av:
                await self._flush_video()
            # Throttled trace of how far the VIDEO trails the VOICE: hold_s = seconds
            # of voice buffered ahead of rendered video. Growing hold_s == the avatar
            # lagging behind the voice (the user's "voice first, avatar later").
            now = asyncio.get_running_loop().time()
            if now - self._last_hold_log >= 1.0:
                self._last_hold_log = now
                hold_s = self._audio_clock_s - self._video_frames / self._fps
                logger.info(
                    f"[ditto sync] hold={hold_s:0.1f}s "
                    f"(audio {self._audio_clock_s:0.1f}s, video {self._video_frames/self._fps:0.1f}s) "
                    f"buf={len(self._abuf)} unsynced={self._unsynced}"
                )
        elif kind == "video_end":
            # Turn's video is done -- flush any remaining frames, then release the tail
            # audio so nothing is stuck. (video first here so trailing closed-mouth frames
            # show before we drain the final audio.)
            if self._sync_av:
                await self._flush_video()
            self._video_active = False
            await self._drain_all()

    async def _fallback_watch(self):
        """If no real video clock advances within fallback_s of the first buffered
        chunk, give up syncing for this turn and forward audio so it never goes
        silent (e.g. a server without the markers)."""
        try:
            await asyncio.sleep(self._fallback_s)
        except asyncio.CancelledError:
            return
        if self._video_frames == 0 and self._abuf:
            logger.warning(
                f"Ditto sync: no video clock within {self._fallback_s}s; "
                f"forwarding voice unsynced for this turn."
            )
            self._unsynced = True
            # Give up sync_with_audio too: push any buffered frames immediately (plain,
            # not audio-locked) and stop buffering, so the avatar isn't stuck on neutral.
            self._video_active = False
            if self._vbuf:
                async with self._lock:
                    pending, self._vbuf = self._vbuf, []
                for img in pending:
                    await self.push_frame(
                        OutputImageRawFrame(image=img, size=self._size, format="RGB"),
                        FrameDirection.DOWNSTREAM,
                    )
            await self._drain_all()

    def _cancel_fallback(self):
        if self._fallback_task:
            self._fallback_task.cancel()
            self._fallback_task = None

    # --- frame processing --------------------------------------------------
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._connect()
            await self.push_frame(frame, direction)

        elif isinstance(frame, (EndFrame, CancelFrame)):
            self._cancel_fallback()
            await self._drain_all()
            await self._disconnect()
            await self.push_frame(frame, direction)

        elif isinstance(frame, InterruptionFrame):
            # Barge-in: reset the server and drop any voice still buffered so the
            # interrupted turn can't keep talking.
            await self._reset_session()
            self._cancel_fallback()
            async with self._lock:
                self._abuf.clear()
            self._reset_turn()
            await self.push_frame(frame, direction)

        elif isinstance(frame, TTSStartedFrame):
            # New turn: reset the shared clock (audio + video start from 0 together)
            # and tell the SERVER to start a speaking segment immediately. The
            # DOWNSTREAM marker is buffered at clock 0 so it releases right before
            # the first audio chunk -- which means BotStartedSpeaking (and TTFO)
            # aligns with the REAL video start, not when TTS was produced.
            self._cancel_fallback()
            self._reset_turn()
            self._turn_active = True
            if self._ws:
                try:
                    await self._ws.send(json.dumps({"type": "speech_start"}))
                except Exception:  # noqa: BLE001
                    pass
            await self._buffer_marker(frame, direction)

        elif isinstance(frame, TTSStoppedFrame):
            # End of TTS for this turn. Tell the server immediately; buffer the
            # downstream marker at the final audio clock so it releases after all
            # the voice has played (the buffered tail drains on video_end/fallback).
            if self._ws:
                try:
                    await self._ws.send(json.dumps({"type": "speech_end"}))
                except Exception:  # noqa: BLE001
                    pass
            self._turn_active = False
            await self._buffer_marker(frame, direction)

        elif isinstance(frame, TTSAudioRawFrame):
            # Send to Ditto for lip-sync IMMEDIATELY (rendering can't wait), but
            # BUFFER the downstream copy -- it's released frame-clocked so the voice
            # the browser hears lines up with the actually-rendered video.
            pcm = _to_16k_mono_pcm(
                frame.audio,
                getattr(frame, "sample_rate", DITTO_SR),
                getattr(frame, "num_channels", 1),
            )
            self._dbg_audio += 1
            if self._dbg_audio == 1:
                # One liveness line at info level: first TTS audio reached the avatar.
                logger.info(
                    f"First TTS audio -> avatar (ws={self._ws is not None}, "
                    f"sr={getattr(frame, 'sample_rate', '?')}, pcm_bytes={len(pcm)})"
                )
            elif self._dbg_audio % 100 == 0:
                logger.debug(f"[ditto] TTSAudio #{self._dbg_audio} pcm_bytes={len(pcm)}")
            if self._ws and pcm:
                try:
                    await self._ws.send(pcm)
                except Exception:  # noqa: BLE001
                    logger.exception("Failed sending audio to Ditto")
            await self._buffer_audio(frame, direction)

        else:
            await self.push_frame(frame, direction)
