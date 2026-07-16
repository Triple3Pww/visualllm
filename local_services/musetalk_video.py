"""MuseTalk local lip-sync avatar as a Pipecat FrameProcessor.

Sits between TTS and transport.output(). It streams the TTS audio to a local MuseTalk
server over a websocket, receives lip-synced RGB frames back, and pushes both video
(OutputImageRawFrame) and audio (TTSAudioRawFrame) downstream IN SYNC.

A/V sync is the hard part. The server renders with some
latency, so we cannot just forward the audio immediately and let the video free-run (that is
the desync). Instead every server frame is SELF-DESCRIBING (proto 2, P51): a 16-byte header
carries kind (real render / held re-send / idle) + audio_pos, the cumulative 16k samples the
server actually consumed rendering it. This client buffers the voice and, per real frame,
releases the audio due by that frame's audio_pos, tagging the frame
`OutputImageRawFrame.sync_with_audio=True` so the transport (non-live) pins each frame to its
audio position -- pairing is the server's own account, so an fps disagreement or a held frame
can never shift the audio<->lip mapping. `video_start`/`video_end` markers still bound the
turn (activation, tail drain, close fade); `video_clock` is diagnostic. A server that never
acks proto 2 trips the loud unsynced fallback. Two strategies (MUSETALK_SYNC_MODE):

  steady    : release incrementally as `video_clock` advances. Because MuseTalk renders
              steadily at ~real-time (no diffusion warmup), the clock advances smoothly so the
              voice plays smoothly -- low latency, no stutter.
  prerender : buffer the whole short reply, release it all aligned on `video_end` -- near-perfect
              sync at the cost of ~one render's worth of extra start delay.

Set MUSETALK_SYNC_WITH_AUDIO=0 to fall back to the old free-running behaviour.
Talks to local_services/musetalk_server/ (FastAPI + websocket). Requires `pip install websockets`.
"""
from __future__ import annotations

import asyncio
import json
import os
import struct

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

MUSETALK_SR = 16000  # Whisper (server-side) expects 16 kHz mono

# proto-2 frame header (mirrors musetalk_server/app.py): magic 4s | kind u8 | 3 pad |
# audio_pos u64 LE. kind 0 = real render, 1 = held re-send, 2 = idle/neutral. audio_pos =
# cumulative 16k samples of the turn's audio covered once the frame shows -- the server's
# own account of what it rendered, which replaces both the i/fps pairing arithmetic and
# the byte-identical held-frame heuristic (P39) when the server speaks proto 2.
FRAME_HDR = struct.Struct("<4sB3xQ")
FRAME_MAGIC = b"MTF2"


def _to_16k_mono_pcm(audio: bytes, in_rate: int, channels: int) -> bytes:
    """Resample int16 PCM to 16 kHz mono for the MuseTalk server."""
    if len(audio) & 1:
        audio = audio[:-1]  # should never fire: run_tts aligns frames at the producer. Kept as
        #                     a crash guard only -- int16 needs an even byte count or np.frombuffer
        #                     raises. (If odd buffers ever reappear, fix the PRODUCER: a dropped
        #                     byte here half-sample-shifts the stream = the P40 noise.)
    a = np.frombuffer(audio, dtype=np.int16)
    if a.size == 0:
        return b""
    if channels and channels > 1:
        a = a.reshape(-1, channels).mean(axis=1)
    if in_rate and in_rate != MUSETALK_SR:
        n_out = int(round(a.shape[0] * MUSETALK_SR / in_rate))
        if n_out <= 0:
            return b""
        src = np.arange(a.shape[0], dtype=np.float64)
        dst = np.linspace(0, a.shape[0] - 1, num=n_out)
        a = np.interp(dst, src, a)
    return a.astype(np.int16).tobytes()


class MuseTalkVideoService(FrameProcessor):
    def __init__(
        self,
        *,
        base_url: str,
        fps: int = 20,
        image_size: tuple[int, int] = (512, 512),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._ws_url = base_url.replace("http", "ws", 1).rstrip("/") + "/stream"
        self._fps = float(fps)
        self._size = image_size
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._recv_task: asyncio.Task | None = None
        self._closing = False

        # --- sync config ---
        # MODE:
        #   live (default) = AUDIO-MASTER. The voice plays at real-time; lip-sync is best-effort
        #     and bounded -- we stop feeding the server when it falls > MAX_LAG behind, so on a
        #     slow/contended GPU the lips skip stale content to stay current instead of dragging
        #     the WHOLE voice slow/late. This is the only sane behaviour when the render can't
        #     sustain real-time (MuseTalk shares the GPU with CosyVoice).
        #   steady / prerender = VIDEO-MASTER (sync_with_audio pinning) -- tight sync ONLY when the
        #     render keeps up; on a slow GPU they make the voice lag. Kept for fast-GPU setups.
        self._mode = (os.getenv("MUSETALK_SYNC_MODE", "steady") or "steady").lower()
        self._sync = self._mode in ("steady", "prerender") and (
            os.getenv("MUSETALK_SYNC_WITH_AUDIO", "1") or "1").lower() in ("1", "true", "yes", "on")
        self._fallback_s = float(os.getenv("MUSETALK_SYNC_FALLBACK_S", "10.0"))
        self._last_hold_log = 0.0

        # Real-time-paced feed to the server (live mode). CosyVoice produces the whole reply
        # FASTER than real-time (RTF<1); if we forwarded all that audio to the renderer as fast
        # as it arrives, the server renders a big backlog that plays out at fps -> the video
        # trails the voice by seconds ("audio done, avatar still going"). So we release audio to
        # the server paced to real-time, keeping the render in lockstep with playback.
        self._feed_q: asyncio.Queue = asyncio.Queue()
        self._feed_task: asyncio.Task | None = None
        # Startup-latency fix: burst the first MUSETALK_FEED_BURST_S of a turn's audio to the
        # server WITHOUT real-time pacing, so it can render the opening frames immediately. The
        # lips otherwise start ~2s late because a fully real-time-paced feed STARVES the renderer
        # at turn start (it can't fill its lead-prime + first segment until audio trickles in).
        # After the burst we resume real-time pacing so no backlog builds (the original guarantee).
        self._burst_s = float(os.getenv("MUSETALK_FEED_BURST_S", "1.0"))
        self._burst_remaining = 0.0

        # --- per-turn sync state ---
        self._lock = asyncio.Lock()
        self._abuf: list[tuple[float, Frame, FrameDirection]] = []  # (cum_end_s, frame, dir)
        self._aidx = 0                # audio release cursor into _abuf
        self._audio_clock_s = 0.0     # seconds of audio buffered this turn
        self._vbuf: list[bytes] = []  # rendered frames this turn (index == real frame #)
        self._vpos: list[int] = []    # proto-2: per-frame audio_pos (16k samples; 0 = unknown)
        self._proto2 = False          # server acked {"type":"proto","v":2} on this ws
        self._released_idx = 0        # video release cursor into _vbuf
        self._video_active = False    # between video_start and video_end
        self._unsynced = False        # fallback engaged this turn
        self._fallback_task: asyncio.Task | None = None
        self._feed_first = None       # first PCM of the turn actually sent ([barge] trace)
        self._flush_t0 = None         # barge-in window open time + frames it discarded ([barge] log)
        self._flush_dropped = 0
        self._flushing = False        # barge-in flush: DROP every incoming server frame until the
        #   next turn's TTSStartedFrame. On an InterruptionFrame the audio queue is drained and the
        #   server told to reset, but frames it already rendered are still in flight on the ws (and
        #   its out_q); without this they arrive with _video_active=False and get forwarded as IDLE
        #   animation -> the avatar keeps lip-moving with NO voice, and any that land after the next
        #   video_start bleed into the new turn. Set True on interrupt, cleared on the next TTSStarted.
        # Held end-of-speech marker (P53, the P11 fix). Under steady the voice is HELD here and
        # released paced to rendered frames, but the TTSStoppedFrame used to sail straight through
        # -- so the transport fired BotStoppedSpeaking MID-turn, the held audio released after it
        # re-fired BotStartedSpeaking, and (with the screech fix's BOT_VAD_STOP_FALLBACK_SECS=600)
        # no BotStopped ever followed: echo-guard's mic mute keyed on it -> mic STUCK MUTED (P11).
        # Fix: hold the stop frame and release it from _drain_audio, AFTER the turn's voice has
        # fully gone downstream, so BotStopped fires exactly once at TRUE end of speech.
        self._held_tts_stop: tuple[Frame, FrameDirection] | None = None

        # --- per-turn A/V timing instrumentation (logs audio-vs-avatar offset + lip drift) ---
        self._t_audio_first: float | None = None   # loop.time() of first voice chunk this turn
        self._t_vid_first: float | None = None      # loop.time() of first rendered frame this turn
        self._t_vid_last: float | None = None
        self._vframes = 0                           # real lip-synced frames this turn
        self._aud_dur = 0.0                         # seconds of voice this turn
        self._last_offset_log = 0.0                 # throttle for the continuous offset trace
        # (The _odd_carry/_srv_carry whole-sample guards that used to live here are GONE: the
        #  P3/P40 odd-byte class is now fixed at its single source -- cosyvoice_tts.run_tts
        #  carries the dangling byte across iter_chunked() reads, so every TTSAudioRawFrame
        #  arrives whole-sample. Verified live 2026-07-15. PROBLEMS-AND-FIXES P3/P40.)
        # EVIDENCE (viseme-mismatch hunt): server re-sends the LAST frame (byte-identical HELD/dup) on
        # every tick it underflows mid-turn; the client can't tell it from a real frame -> it lands in
        # _vbuf and shifts the audio<->lip mapping. Count held dups vs the server's REAL rendered count.
        self._held_dups = 0
        self._server_real = 0

        # --- smooth end-of-turn close (steady) ---
        # MuseTalk can't ease the mouth shut itself (silence renders a PARTED mouth, not closed
        # lips -- measured), so at end of turn we cross-dissolve the last spoken frame -> the rest
        # pose over K frames. Those frames are pushed FREE-RUN (untagged, like the idle loop) by
        # _play_close_fade, NOT paired with silence through _emit_pair -- the audio-cap in _advance
        # would strand them whenever the render fell behind the voice. Gated by
        # MUSETALK_CLOSE_FADE_FRAMES (0 = off, the old clean snap). Use with
        # MUSETALK_END_TAIL_FRAMES=0 so the last buffered frame is the last SPOKEN frame, not a
        # neutral tail copy.
        self._close_fade = int(os.getenv("MUSETALK_CLOSE_FADE_FRAMES", "0") or "0")
        self._rest_frame: bytes | None = None   # cached between-turn rest pose (crossfade target)
        self._suppress_until = 0.0              # drop server idle frames until here so they can't
        #   preempt the crossfade playout (the burst-flush collapse P12 hit)

        # --- freeze watchdog (capture the REAL freeze the ~1s hold/offset sampling misses) ---
        # A freeze = video frames stop reaching the transport for a beat. Track the wall-gap
        # between EMITTED OutputImageRawFrames (every release path funnels through push_frame) and
        # warn the instant it exceeds MUSETALK_STALL_LOG_S, classified render-starved (the server
        # also stopped feeding us -> high arrival gap) vs delivery-side (frames buffered but not
        # going out). A TOTAL stall emits nothing, so a poller (_watch_loop) raises it rather than
        # the next emit. Pairs with the browser-side monitor in main.py for the transport/browser
        # leg the server can't see. Default 0 (OFF) -- diagnostic scaffolding; set a value like
        # 0.4 to re-arm it when hunting a freeze.
        self._stall_s = float(os.getenv("MUSETALK_STALL_LOG_S", "0") or "0")
        self._last_emit_t: float | None = None   # loop.time() of the last video frame pushed out
        self._stall_open = False                 # a freeze is currently being reported
        self._watch_task: asyncio.Task | None = None
        # EVIDENCE (#1 audio-content): dump the EXACT 16k PCM this turn sent to the avatar server,
        # so it can be rendered offline (GPU-alone) and compared -- separates a corrupt-audio cause
        # from the pacing cause. Gated by MUSETALK_DUMP_PCM=1 (writes output/_live_turn_pcm.wav).
        self._dump_pcm = (os.getenv("MUSETALK_DUMP_PCM", "0") or "0").lower() in ("1", "true", "yes")
        self._pcm_dump = bytearray()
        # EVIDENCE (fix verification): dump the EXACT A/V the client delivers DOWNSTREAM this turn
        # (the frames + voice the transport/browser actually gets), so the fix can be watched, not just
        # trusted from logs. Gated by MUSETALK_DUMP_DELIVERED=1 -> output/_delivered_{rgb.bin,voice.wav}
        # + _delivered_meta.json; a scratchpad muxer builds the mp4. Off by default.
        self._dump_deliv = (os.getenv("MUSETALK_DUMP_DELIVERED", "0") or "0").lower() in ("1", "true", "yes")
        self._deliv_active = False
        self._deliv_v: list[bytes] = []          # delivered RGB frame buffers, in order
        self._deliv_a = bytearray()              # delivered voice PCM (downstream sample rate)
        self._deliv_sr = MUSETALK_SR
        self._deliv_ch = 1

    def _write_deliv_dump(self):
        """Write the turn's exact delivered frames + voice for offline muxing (evidence, best-effort)."""
        import json as _json
        try:
            n = len(self._deliv_v)
            side = int(round((len(self._deliv_v[0]) // 3) ** 0.5)) if n else 0
            with open(os.path.join("output", "_delivered_rgb.bin"), "wb") as f:
                for fr in self._deliv_v:
                    f.write(fr)
            import wave
            with wave.open(os.path.join("output", "_delivered_voice.wav"), "wb") as w:
                w.setnchannels(self._deliv_ch); w.setsampwidth(2); w.setframerate(self._deliv_sr)
                w.writeframes(bytes(self._deliv_a))
            with open(os.path.join("output", "_delivered_meta.json"), "w") as f:
                _json.dump({"frames": n, "side": side, "fps": self._fps,
                            "voice_sr": self._deliv_sr, "voice_ch": self._deliv_ch,
                            "voice_s": len(self._deliv_a) / 2 / self._deliv_ch / self._deliv_sr}, f)
            logger.info(f"[deliv-dump] {n} frames ({side}px) + "
                        f"{len(self._deliv_a)/2/self._deliv_ch/self._deliv_sr:0.2f}s voice -> output/_delivered_*")
        except Exception:  # noqa: BLE001
            logger.exception("[deliv-dump] failed")

    def _write_pcm_dump(self):
        """Write the turn's exact server-bound 16k mono PCM to a wav (evidence, best-effort)."""
        import wave
        try:
            out = os.path.join("output", "_live_turn_pcm.wav")
            with wave.open(out, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(MUSETALK_SR)
                w.writeframes(bytes(self._pcm_dump))
            logger.info(f"[pcm-dump] wrote {len(self._pcm_dump)//2} samples "
                        f"({len(self._pcm_dump)/2/MUSETALK_SR:0.2f}s) -> {out}")
        except Exception:  # noqa: BLE001
            logger.exception("[pcm-dump] failed")

    # --- connection lifecycle ---------------------------------------------
    async def _connect(self):
        if self._recv_task is not None:
            return
        self._closing = False
        try:
            await self._open_ws()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"MuseTalk initial connect failed ({e!r}); loop will retry.")
        self._recv_task = asyncio.create_task(self._receive_loop())
        if self._feed_task is None:
            self._feed_task = asyncio.create_task(self._feed_loop())
        if self._watch_task is None and self._stall_s > 0:
            self._watch_task = asyncio.create_task(self._watch_loop())

    async def _open_ws(self):
        logger.info(f"Connecting to MuseTalk server at {self._ws_url} "
                    f"(sync={'on:'+self._mode if self._sync else 'off'})")
        self._ws = await websockets.connect(
            self._ws_url, max_size=None, ping_interval=None, close_timeout=1
        )
        # Request proto 2 (self-describing frames). _proto2 flips on the server's ack
        # marker only, and resets here per connection. Synced mode REQUIRES proto 2 now
        # (the index/fps pairing was deleted, P51): an older server that never acks trips
        # the loud _unsynced fallback in _on_frame instead of silently mis-pairing.
        self._proto2 = False
        await self._ws.send(json.dumps({"type": "config", "fps": self._fps, "proto": 2}))

    async def _feed_loop(self):
        """Send queued items to the server, pacing AUDIO to real-time so the renderer never
        builds a backlog (the cause of the voice-finishes-but-video-keeps-going lag). Markers
        (start/end/reset) are forwarded immediately, in order with the audio."""
        while not self._closing:
            try:
                kind, payload = await self._feed_q.get()
            except asyncio.CancelledError:
                break
            try:
                if self._ws is None:
                    continue
                if kind == "audio":
                    pcm, dur = payload
                    await self._ws.send(pcm)
                    # Split the turn-start delay at its ONE ambiguous seam: did WE hold the audio
                    # (burst budget / pacing / queue backlog), or did the server hold the frames?
                    # The profiler already showed the GPU flat and a ~1.9s gap between segments on
                    # a spiked turn, i.e. the renderer was STARVED -- this says who starved it.
                    if self._feed_first is None:
                        self._feed_first = _t = asyncio.get_running_loop().time()
                        if self._t_audio_first is not None:
                            logger.info(
                                f"[barge] first PCM on the wire +{(_t - self._t_audio_first)*1000:.0f}ms "
                                f"after the turn's first TTS chunk (burst_left={self._burst_remaining:.2f}s "
                                f"qdepth={self._feed_q.qsize()})")
                    if self._burst_remaining > 0:
                        self._burst_remaining -= dur   # BURST: skip the pace so the renderer can
                        #   start the opening frames immediately (kills the ~2s startup starve)
                    else:
                        await asyncio.sleep(dur)        # then pace to real-time (no backlog)
                else:
                    if kind == "speech_start":
                        self._burst_remaining = self._burst_s   # reset the burst budget per turn
                        self._feed_first = None                 # re-arm the per-turn feed trace
                    await self._ws.send(json.dumps({"type": kind}))
            except asyncio.CancelledError:
                break
            except Exception:  # noqa: BLE001
                pass

    async def _disconnect(self):
        self._closing = True
        self._cancel_fallback()
        for task_attr in ("_recv_task", "_feed_task", "_watch_task"):
            task = getattr(self, task_attr)
            if task:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
                setattr(self, task_attr, None)
        if self._ws:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001
                pass
            self._ws = None

    async def _receive_loop(self):
        while not self._closing:
            try:
                if self._ws is None:
                    await self._open_ws()
                assert self._ws is not None
                async for message in self._ws:
                    if isinstance(message, bytes):
                        await self._on_frame(message)
                    else:
                        await self._on_marker(json.loads(message))
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001
                if not self._closing:
                    logger.warning(f"MuseTalk ws dropped ({e!r}); reconnecting...")
            finally:
                ws, self._ws = self._ws, None
                if ws is not None:
                    try:
                        await ws.close()
                    except Exception:  # noqa: BLE001
                        pass
            if not self._closing:
                await asyncio.sleep(0.5)

    # --- sync core --------------------------------------------------------
    def _img_size(self, img: bytes) -> tuple[int, int]:
        """Derive the (square) frame size from the RGB byte length so every OutputImageRawFrame
        SELF-DESCRIBES its true dimensions. A fixed self._size crashes aiortc's VP8 sender
        ('cannot reshape array of size N into (S,S,3)') the instant the avatar server and this
        pipeline briefly disagree on frame size -- e.g. during a MUSETALK_SPLIT toggle, when the
        server may still emit a 512 frame while the pipeline is configured for the 256 crop (or
        vice-versa). That ValueError kills the video RTP task -> the avatar FREEZES for the rest of
        the session (audio is a separate task and keeps working -> 'voice but no moving avatar').
        Sizing from the bytes turns any such mismatch into a harmless transient, never a freeze."""
        n = len(img) // 3
        side = int(round(n ** 0.5))
        return (side, side) if side > 0 and side * side == n else self._size

    async def _on_frame(self, img: bytes):
        """A rendered RGB frame from the server."""
        # proto 2: every binary message is header + pixels. kind/pos make the frame
        # self-describing; kind stays None on a bare (proto-1) frame, which synced mode
        # treats as a protocol error (the loud fallback below) -- the guessing heuristics
        # that used to cover bare frames were deleted with the index/fps pairing (P51).
        kind, pos = None, 0
        if self._proto2 and len(img) >= FRAME_HDR.size and img[:4] == FRAME_MAGIC:
            _m, kind, pos = FRAME_HDR.unpack(img[:FRAME_HDR.size])
            img = img[FRAME_HDR.size:]
        if self._flushing:
            # Post-interrupt: an old-turn frame still draining from the server (ws + out_q). Drop it
            # entirely -- do not count it, buffer it, or forward it as idle. Cleared on next TTSStarted.
            self._flush_dropped += 1
            return
        # HELD/dup detection: the server re-sends the last frame during a render underflow (or
        # the lead-prime). A held frame is not new lip motion, so it must neither count as
        # delivered video nor land in the synced buffer. proto 2 DECLARES it (kind 1/2 = not a
        # new real frame) -- the old byte-compare-with-_vbuf[-1] heuristic (P39) was retired with
        # the index/fps pairing when proto 2 became required for synced mode (P51).
        is_dup = (self._sync and self._video_active and not self._unsynced
                  and kind is not None and kind != 0)
        if (kind is None and self._sync and self._video_active and not self._unsynced):
            # Bare frame inside a synced turn = the server never acked proto 2 (an older build).
            # Synced pairing REQUIRES the per-frame audio_pos now, so fail LOUDLY into the same
            # unsynced fallback _fallback_watch uses (voice forwarded immediately, frames
            # free-run) instead of silently mis-pairing.
            logger.warning("MuseTalk server sent a bare frame (no proto-2 header); synced "
                           "pairing needs proto 2 -- forwarding this turn unsynced.")
            self._unsynced = True
            await self._drain_audio()
        if self._video_active and not is_dup:   # count only REAL frames + trace the offset
            now = asyncio.get_running_loop().time()
            if self._t_vid_first is None:
                self._t_vid_first = now
                # First real rendered frame of the turn. Log the intra-avatar render latency
                # (first voice chunk -> first frame back) so the mic-to-ear waterfall has a real
                # 'Avatar render' anchor instead of bundling it into the steady lead-hold.
                if self._t_audio_first is not None:
                    logger.info(f"[render] first-frame +{(now - self._t_audio_first) * 1000:0.0f}ms")
            self._t_vid_last = now
            self._vframes += 1
            # Continuous (delivery-side) offset: real-time voice elapsed vs lip-video delivered.
            # + = lips behind the voice, - = lips ahead. Shows the swing within a turn (the burst
            # can push the lips AHEAD after they start behind). NOTE: this is delivery to the
            # transport, not browser playout (the jitter buffer smooths some of it).
            if self._t_audio_first is not None and now - self._last_offset_log >= 1.0:
                self._last_offset_log = now
                voice_s = now - self._t_audio_first
                video_s = self._vframes / self._fps
                off = voice_s - video_s
                tag = f"{off:+0.2f}s behind" if off > 0.05 else (
                    f"{-off:0.2f}s AHEAD" if off < -0.05 else "in step")
                logger.info(f"[avatar offset] {voice_s:0.1f}s in: lips {tag}")
        if not self._sync:
            await self.push_frame(
                OutputImageRawFrame(image=img, size=self._img_size(img), format="RGB"),
                FrameDirection.DOWNSTREAM,
            )
            return
        if self._video_active and not self._unsynced:
            async with self._lock:
                if is_dup:
                    # The server re-sends the LAST frame to keep the WebRTC track alive whenever
                    # its render underflows mid-turn (GPU contention with CosyVoice, or a real-time
                    # feed briefly starving it). A held frame landing in _vbuf would be a PHANTOM
                    # real frame: it pairs voice with lip motion that never happened and delivers
                    # every following real frame late -- the live "lips don't match the words" that
                    # offline prerender (GPU-alone, no underflow) never shows (P39). DROP it so
                    # _vbuf stays exactly the REAL rendered sequence; under a stall the client then
                    # holds the last real frame and the voice pauses IN SYNC (the intended steady
                    # tradeoff) instead of drifting. Detection is the wire's own kind flag (held /
                    # idle), not a guess -- the byte-compare heuristic went with proto 1 (P51).
                    self._held_dups += 1
                else:
                    self._vbuf.append(img)
                    self._vpos.append(pos)   # kind-0 frames always carry a real pos (>0)
            if self._mode == "steady":
                await self._advance()   # release paced to frames AS they arrive (continuous)
        else:
            # idle frame (between turns) or fallback: animate immediately, untagged.
            if not self._video_active:
                # Cache the rest pose (crossfade target) and, while a close crossfade is playing
                # out, DROP these server idle frames so they can't preempt it (the burst-flush
                # collapse P12 hit -- the transport's current image would jump to neutral).
                self._rest_frame = img
                if asyncio.get_running_loop().time() < self._suppress_until:
                    return
            await self.push_frame(
                OutputImageRawFrame(image=img, size=self._img_size(img), format="RGB"),
                FrameDirection.DOWNSTREAM,
            )

    async def _on_marker(self, evt: dict):
        kind = evt.get("type")
        if kind == "proto":
            self._proto2 = int(evt.get("v", 1)) >= 2
            logger.info(f"MuseTalk server speaks proto {evt.get('v', 1)} "
                        f"(per-frame audio_pos pairing {'ON' if self._proto2 else 'off'}).")
        elif kind == "video_start":
            # New turn segment. TTSStartedFrame already reset the turn (audio + buffers) BEFORE
            # any audio was buffered, so here we ONLY mark active. We deliberately do NOT clear
            # _vbuf / _released_idx -- they stay continuous across the whole turn so a stray
            # mid-reply re-segment can't desync the frame<->audio mapping.
            self._cancel_fallback()
            self._video_active = True
            if self._dump_deliv:
                self._deliv_active = True   # begin capturing the real speaking segment (A/V aligned)
        elif kind == "video_clock":
            # Diagnostic only since proto 2 (P51): pairing rides on each frame's audio_pos, and
            # release fires on frame receipt. Under proto 1 this marker ALSO re-ran _advance to
            # un-stick frames the audio-cap had parked until more voice arrived -- both the cap
            # and that heartbeat are gone (a proto-2 pos can never exceed audio already buffered).
            self._server_real = int(evt.get("frames", self._server_real))  # server's REAL frame count
        elif kind == "video_end":
            close_start = self._vbuf[-1] if self._vbuf else None   # last spoken frame (crossfade src)
            await self._advance()       # flush whatever is buffered (prerender: the whole reply)
            await self._drain_audio()   # release the turn's trailing voice
            self._log_turn_timing()
            if self._dump_deliv and self._deliv_active:
                self._deliv_active = False
                self._write_deliv_dump()
            self._video_active = False
            if self._close_fade > 0 and close_start is not None and self._rest_frame is not None:
                # Ease the mouth shut: FREE-RUN the crossfade (untagged, like the idle loop) so it
                # is NOT gated by the audio-cap in _advance -- that cap strands trailing frames when
                # the render ran behind (video > audio). Suppress server idle frames during the
                # playout so they can't preempt it; the fade lands on the rest pose, so the neutral
                # the server then holds is seamless. ("Live during the close" within steady.)
                self._suppress_until = (asyncio.get_running_loop().time()
                                        + self._close_fade / self._fps + 0.3)
                asyncio.create_task(self._play_close_fade(close_start, self._rest_frame))

    async def _advance(self):
        """Release received frames, each paired (in order) with the audio due by its audio_pos
        and tagged sync_with_audio so the transport pins it. No audio-cap is needed (the old
        P10 ceil-cap guarded the index/fps pairing): a real frame's audio_pos can never exceed
        voice the client already buffered -- _abuf.append happens in the same process_frame call
        that queues the server feed, and the server renders only from audio it was fed -- so a
        pos-paired release can't run ahead of the voice by construction (P51)."""
        async with self._lock:
            while self._released_idx < len(self._vbuf):
                await self._emit_pair(self._released_idx)
                self._released_idx += 1
            self._log_hold()

    async def _emit_pair(self, i: int):
        """Audio due by frame i's audio_pos (in order), then frame i tagged sync_with_audio.
        Caller holds the lock. The frame's own audio_pos -- what the server ACTUALLY consumed
        rendering it -- defines the audio due; the old i/fps index arithmetic went with proto 1
        (P51), so an fps disagreement can't shift the mapping."""
        ft = self._vpos[i] / MUSETALK_SR
        while self._aidx < len(self._abuf) and self._abuf[self._aidx][0] <= ft:
            _e, af, ad = self._abuf[self._aidx]
            self._aidx += 1
            await self.push_frame(af, ad)
        if i < len(self._vbuf):
            fr = OutputImageRawFrame(image=self._vbuf[i], size=self._img_size(self._vbuf[i]), format="RGB")
            fr.sync_with_audio = True
            await self.push_frame(fr, FrameDirection.DOWNSTREAM)

    def _log_hold(self):
        now = asyncio.get_running_loop().time()
        if now - self._last_hold_log >= 1.0:
            self._last_hold_log = now
            hold = self._audio_clock_s - self._released_idx / self._fps
            logger.info(
                f"[musetalk sync] hold={hold:0.2f}s (audio {self._audio_clock_s:0.1f}s, "
                f"video {self._released_idx/self._fps:0.1f}s) "
                f"abuf={len(self._abuf)-self._aidx} vbuf={len(self._vbuf)-self._released_idx}"
            )

    async def _drain_audio(self):
        """Release any audio left after the last rendered frame (tail of the turn), then the
        held TTSStoppedFrame (P53): it must reach the transport AFTER the turn's voice so
        BotStoppedSpeaking fires at true end of speech, not mid-turn (the P11 stuck-mute).
        Every turn-drain path funnels here (video_end, the marker-loss fallback, the proto-1
        bare-frame fallback), so the release can't be missed."""
        async with self._lock:
            while self._aidx < len(self._abuf):
                _e, af, ad = self._abuf[self._aidx]
                self._aidx += 1
                await self.push_frame(af, ad)
            if self._held_tts_stop is not None:
                sf, sd = self._held_tts_stop
                self._held_tts_stop = None
                await self.push_frame(sf, sd)

    async def _play_close_fade(self, last_bytes: bytes, rest_bytes: bytes):
        """Free-run a pixel cross-dissolve (last spoken frame -> rest pose) at fps so the mouth
        eases shut at end of turn. Untagged frames (like the idle loop) so the non-live transport
        draws them on its own clock -- NOT audio-paired, so the _advance audio-cap can't strand
        them when the render fell behind (video > audio). We blend PIXELS because MuseTalk renders
        silence as a PARTED mouth, not closed (measured), so feeding it can't close the mouth. The
        final blended frame == the rest pose, so the neutral the server holds afterwards is seamless.
        Best-effort: any failure just leaves the prior clean snap."""
        try:
            last = np.frombuffer(last_bytes, dtype=np.uint8).astype(np.float32)
            rest = np.frombuffer(rest_bytes, dtype=np.uint8).astype(np.float32)
            if last.shape != rest.shape:
                return
            interval = 1.0 / self._fps
            for j in range(1, self._close_fade + 1):
                a = j / self._close_fade                    # linear: lands exactly on the rest pose
                blended = ((1.0 - a) * last + a * rest).astype(np.uint8).tobytes()
                await self.push_frame(
                    OutputImageRawFrame(image=blended, size=self._img_size(blended), format="RGB"),
                    FrameDirection.DOWNSTREAM)
                await asyncio.sleep(interval)
        except Exception:  # noqa: BLE001 -- close polish only; never disrupt the next turn
            pass

    def _reset_turn(self):
        self._abuf = []
        self._aidx = 0
        self._audio_clock_s = 0.0
        self._vbuf = []
        self._vpos = []
        self._released_idx = 0
        self._video_active = False
        self._unsynced = False
        self._t_audio_first = None
        self._t_vid_first = None
        self._t_vid_last = None
        self._vframes = 0
        self._aud_dur = 0.0
        self._last_offset_log = 0.0
        self._held_dups = 0
        self._server_real = 0
        # Drop a still-held stop marker (P53). Reached on InterruptionFrame -- correct to DROP
        # there: pipecat's transport emits its own BotStopped on interruption, and forwarding
        # ours later would fire a bogus mid-turn BotStopped into the NEXT turn. The TTSStarted
        # path flushes it BEFORE calling this, so a normal turn never loses its stop here.
        self._held_tts_stop = None

    def _log_turn_timing(self):
        """Log this turn's audio-vs-avatar timing: how long after the voice the lips started,
        and how far the video fell behind by the end (the live lip drift). Best-effort."""
        if self._t_vid_first is None or self._t_audio_first is None:
            return
        startup = self._t_vid_first - self._t_audio_first
        vid_dur = self._vframes / self._fps if self._fps else 0.0
        span = (self._t_vid_last - self._t_vid_first) if self._t_vid_last else 0.0
        eff_fps = self._vframes / span if span > 0 else 0.0
        drift = self._aud_dur - vid_dur
        # The perceived lip lag is dominated by how late the lips START after the voice
        # (startup), plus any accumulating drift over the turn. Both must be small to be in step.
        verdict = "LIPS BEHIND" if (startup > 0.15 or drift > 0.15) else "in step"
        logger.info(
            f"[avatar timing] lips start +{startup:0.2f}s after voice | "
            f"audio {self._aud_dur:0.2f}s video {vid_dur:0.2f}s "
            f"({self._vframes} frames, {eff_fps:0.1f} fps) | "
            f"end drift +{drift:0.2f}s -> {verdict} | "
            f"held/dup {self._held_dups} of {self._vframes} recv (server-real {self._server_real})"
        )

    # --- fallback (marker-less server / lost markers) ---------------------
    def _arm_fallback(self):
        if self._fallback_task or not self._sync:
            return
        self._fallback_task = asyncio.create_task(self._fallback_watch())

    def _cancel_fallback(self):
        if self._fallback_task:
            self._fallback_task.cancel()
            self._fallback_task = None

    async def _fallback_watch(self):
        try:
            await asyncio.sleep(self._fallback_s)
        except asyncio.CancelledError:
            return
        if not self._video_active and self._abuf and self._aidx < len(self._abuf):
            logger.warning(f"MuseTalk sync: no video markers within {self._fallback_s}s; "
                           "forwarding voice unsynced for this turn.")
            self._unsynced = True
            await self._drain_audio()

    async def _reset_session(self):
        if self._ws:
            try:
                await self._ws.send(json.dumps({"type": "reset"}))
            except Exception:  # noqa: BLE001
                pass

    # (The _align_even() whole-sample guard that used to sit on push_frame is GONE -- the odd
    #  bytes it repaired are fixed at their source now (cosyvoice_tts.run_tts carries the byte
    #  across iter_chunked() reads), so the pipecat transport's audio buffer total stays even
    #  and its discard-on-stall can never half-sample-shift the stream. P3 anti-screech history:
    #  docs/PROBLEMS-AND-FIXES.md P3; the BOT_VAD_STOP_FALLBACK_SECS raise in main.py STAYS.)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, OutputImageRawFrame):
            self._note_emit()         # freeze watchdog: a video frame is leaving downstream
        if self._dump_deliv and self._deliv_active and direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, OutputImageRawFrame):
                self._deliv_v.append(frame.image)   # exact delivered frame, in order
            elif isinstance(frame, TTSAudioRawFrame):
                self._deliv_sr = getattr(frame, "sample_rate", self._deliv_sr) or self._deliv_sr
                self._deliv_ch = getattr(frame, "num_channels", self._deliv_ch) or self._deliv_ch
                self._deliv_a.extend(frame.audio)   # exact delivered voice
        await super().push_frame(frame, direction)

    def _note_emit(self):
        """Mark a video frame's departure; close out any freeze the watchdog opened (the gap
        since the last emit IS the freeze duration, so log it before overwriting the timestamp)."""
        if self._stall_s <= 0:
            return
        now = asyncio.get_running_loop().time()
        if self._stall_open and self._last_emit_t is not None:
            logger.warning(f"[avatar FREEZE] recovered after {(now - self._last_emit_t) * 1000:.0f}ms")
            self._stall_open = False
        self._last_emit_t = now

    async def _watch_loop(self):
        """Poll for a video-out stall the ~1s hold/offset logs can't see: if no frame has gone
        downstream for MUSETALK_STALL_LOG_S while a turn is live (or its audio still draining),
        log it ONCE with the state that localizes it (render-starved vs delivery-side)."""
        while not self._closing:
            await asyncio.sleep(0.2)
            if self._last_emit_t is None or self._stall_open:
                continue
            if not (self._video_active or self._aidx < len(self._abuf)):
                continue   # between turns, holding the rest pose is not a freeze
            now = asyncio.get_running_loop().time()
            gap = now - self._last_emit_t
            if gap < self._stall_s:
                continue
            arr = (now - self._t_vid_last) if self._t_vid_last is not None else -1.0
            cause = ("render-starved (server not sending frames)" if arr >= self._stall_s
                     else "delivery-side (frames buffered, not going downstream)")
            logger.warning(
                f"[avatar FREEZE] no video out for {gap * 1000:.0f}ms -> {cause}; "
                f"server-arrival-gap={arr * 1000:.0f}ms "
                f"vbuf={len(self._vbuf) - self._released_idx} abuf={len(self._abuf) - self._aidx} "
                f"active={self._video_active}"
            )
            self._stall_open = True

    # --- frame processing --------------------------------------------------
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._connect()
            await self.push_frame(frame, direction)

        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._disconnect()
            await self.push_frame(frame, direction)

        elif isinstance(frame, InterruptionFrame):
            # Barge-in: drop any audio still queued for the server so the interrupted turn
            # can't keep driving the lips, then reset.
            while not self._feed_q.empty():
                try:
                    self._feed_q.get_nowait()
                except asyncio.QueueEmpty:
                    break
            await self._reset_session()
            self._cancel_fallback()
            self._reset_turn()
            self._flushing = True   # drop in-flight server frames until the next turn starts
            # Barge-in is the BASELINE (ALLOW_INTERRUPTIONS=1) and the whole path was silent, so a
            # barge-in that costs the user latency looked identical to one that did not. Measured
            # 2026-07-16: ~1 in 4 close-following turns has its first frame land +1.7s late while
            # the GPU stays flat (per-segment cost identical), i.e. the server renders on time and
            # THIS flush window drops the frames. These 2 lines make the window's cost visible.
            self._flush_t0 = asyncio.get_running_loop().time()
            self._flush_dropped = 0
            logger.info("[barge] interrupt -> flush ON (dropping server frames until next TTSStarted)")
            await self.push_frame(frame, direction)

        elif isinstance(frame, TTSAudioRawFrame):
            sr = getattr(frame, "sample_rate", MUSETALK_SR) or MUSETALK_SR
            ch = getattr(frame, "num_channels", 1) or 1
            dur = (len(frame.audio) // (2 * ch)) / sr
            if self._t_audio_first is None:   # first voice chunk of the turn = audio start
                self._t_audio_first = asyncio.get_running_loop().time()
            self._aud_dur += dur
            # ALWAYS feed the server REAL-TIME-PACED (via _feed_q) so the renderer can't build a
            # backlog from CosyVoice's faster-than-real-time output (the "voice finishes but the
            # avatar keeps going" lag). Pacing keeps the server's queue ~empty either mode.
            # Frames arrive whole-sample: cosyvoice_tts.run_tts aligns at the producer (the P40
            # "avatar fed NOISE, mouth flaps wordlessly" bug was odd-byte frames + a dropped
            # stray byte here; fixed at the source 2026-07-15, the _srv_carry patch removed).
            data = frame.audio
            pcm = _to_16k_mono_pcm(data, sr, ch) if data else b""
            if pcm:
                self._feed_q.put_nowait(("audio", (pcm, dur)))
                if self._dump_pcm:
                    self._pcm_dump.extend(pcm)   # exact server-bound PCM (evidence)
            if not self._sync:
                # AUDIO-MASTER (live): forward the voice NOW (plays at real-time, lips best-effort).
                await self.push_frame(frame, direction)
            elif self._unsynced:
                await self.push_frame(frame, direction)   # fallback: marker-less server
            else:
                # READINESS-GATED (steady): hold the voice and release it locked to the real
                # rendered frames -- the voice waits until the avatar is ready, then they play
                # together. No drift, no end cut. (See _advance / _emit_pair: each frame's
                # audio_pos defines the audio released with it.)
                # The mid-speech "screech" that steady used to hit is NOT a held-frame problem --
                # it was pipecat discarding a partial (odd) audio buffer; fixed at the source
                # (run_tts keeps every frame whole-sample) + the BOT_VAD_STOP_FALLBACK_SECS raise
                # in main.py. So we just buffer the frame as-is here.
                self._audio_clock_s += dur
                async with self._lock:
                    self._abuf.append((self._audio_clock_s, frame, direction))
                self._arm_fallback()

        elif isinstance(frame, TTSStartedFrame):
            self._cancel_fallback()
            if self._held_tts_stop is not None:
                # Safety net (P53): the previous turn's video_end never came (server died / ws
                # dropped mid-turn), so its held stop was never drained. Release it BEFORE the
                # new turn so the transport still closes the old turn (BotStopped) and the
                # upcoming audio opens a fresh one -- otherwise echo-guard's mute would stick.
                sf, sd = self._held_tts_stop
                self._held_tts_stop = None
                await self.push_frame(sf, sd)
            self._reset_turn()
            if self._flushing:   # close the barge-in window: how long, and how much it discarded
                held = asyncio.get_running_loop().time() - (self._flush_t0 or 0.0)
                logger.info(f"[barge] flush OFF at TTSStarted: window={held:.2f}s "
                            f"dropped={self._flush_dropped} server frames")
            self._flushing = False   # new turn: stop dropping frames (end the post-interrupt flush)
            if self._dump_pcm:
                self._pcm_dump = bytearray()
            if self._dump_deliv:
                self._deliv_active = False   # capture starts at video_start (skip the pre-speech neutral hold)
                self._deliv_v = []
                self._deliv_a = bytearray()
            # speech_start/end go through _feed_q so they order correctly with the real-time-
            # paced audio (start before the turn's audio, end after it fully drains).
            self._feed_q.put_nowait(("speech_start", None))
            await self.push_frame(frame, direction)

        elif isinstance(frame, TTSStoppedFrame):
            self._feed_q.put_nowait(("speech_end", None))
            if self._dump_pcm and self._pcm_dump:
                self._write_pcm_dump()
            # P53 (the P11 fix): under steady the voice is still HELD in _abuf when this
            # arrives, so forwarding it now makes the transport fire BotStopped mid-turn and
            # the later-released audio re-fire BotStarted with nothing to close it (the
            # echo-guard stuck-mute). Hold it; _drain_audio pushes it after the last voice.
            held = False
            if self._sync and not self._unsynced:
                async with self._lock:
                    if self._video_active or self._aidx < len(self._abuf):
                        self._held_tts_stop = (frame, direction)
                        held = True
            if not held:
                await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)
