"""MuseTalk local lip-sync avatar as a Pipecat FrameProcessor.

Sits between TTS and transport.output(). It streams the TTS audio to a local
MuseTalk server over a websocket, receives lip-synced RGB video frames back, and
pushes both the video (OutputImageRawFrame) and the audio (TTSAudioRawFrame)
downstream so the browser plays them in sync.

Design mirrors Pipecat's hosted avatar services (Simli/HeyGen):
- Audio is forwarded downstream *immediately* so speech is never blocked on
  frame generation.
- Video frames stream from the server at a fixed FPS (default 25) and are
  emitted as they arrive — first frame defines the avatar's TTFO contribution.
- On interruption (barge-in) we reset the server session so stale frames don't
  leak into the next turn.

Talks to local_services/musetalk_server/ (FastAPI + websocket).
Requires: `pip install websockets`.
"""
from __future__ import annotations

import asyncio
import json

import numpy as np
import websockets
from loguru import logger

MUSETALK_SR = 16000  # Whisper (server-side) expects 16 kHz mono


def _to_16k_mono_pcm(audio: bytes, in_rate: int, channels: int) -> bytes:
    """Resample int16 PCM to 16 kHz mono for the MuseTalk server.

    MuseTalk's Whisper encoder requires 16 kHz; TTS often emits 24 kHz. Per-chunk
    linear resampling is sufficient for lip-sync feature extraction.
    """
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

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    OutputImageRawFrame,
    StartFrame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class MuseTalkVideoService(FrameProcessor):
    def __init__(
        self,
        *,
        base_url: str,
        fps: int = 20,  # match the server's realtime budget on the 5060 Ti
        image_size: tuple[int, int] = (512, 512),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._ws_url = base_url.replace("http", "ws", 1).rstrip("/") + "/stream"
        self._fps = fps
        self._size = image_size
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._recv_task: asyncio.Task | None = None

    # --- connection lifecycle ---------------------------------------------
    async def _connect(self):
        if self._ws is not None:
            return
        logger.info(f"Connecting to MuseTalk server at {self._ws_url}")
        self._ws = await websockets.connect(self._ws_url, max_size=None)
        await self._ws.send(json.dumps({"type": "config", "fps": self._fps}))
        self._recv_task = asyncio.create_task(self._receive_frames())

    async def _disconnect(self):
        if self._recv_task:
            self._recv_task.cancel()
            self._recv_task = None
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def _receive_frames(self):
        """Read RGB frames from the server and push them downstream."""
        try:
            assert self._ws is not None
            async for message in self._ws:
                if isinstance(message, bytes):
                    # Raw RGB frame buffer (image_size * 3 bytes).
                    await self.push_frame(
                        OutputImageRawFrame(
                            image=message, size=self._size, format="RGB"
                        ),
                        FrameDirection.DOWNSTREAM,
                    )
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001
            logger.exception("MuseTalk frame receiver stopped")

    async def _reset_session(self):
        """Flush server-side buffers on interruption."""
        if self._ws:
            try:
                await self._ws.send(json.dumps({"type": "reset"}))
            except Exception:  # noqa: BLE001
                pass

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
            await self._reset_session()
            await self.push_frame(frame, direction)

        elif isinstance(frame, TTSAudioRawFrame):
            # Feed audio to MuseTalk for lip-sync, AND forward it downstream so
            # speech plays without waiting on frame generation. The server's
            # Whisper needs 16 kHz mono, so resample before sending (the frame
            # forwarded downstream keeps its original rate for playback).
            if self._ws:
                try:
                    pcm = _to_16k_mono_pcm(
                        frame.audio,
                        getattr(frame, "sample_rate", MUSETALK_SR),
                        getattr(frame, "num_channels", 1),
                    )
                    if pcm:
                        await self._ws.send(pcm)
                except Exception:  # noqa: BLE001
                    logger.exception("Failed sending audio to MuseTalk")
            await self.push_frame(frame, direction)

        elif isinstance(frame, (TTSStartedFrame, TTSStoppedFrame)):
            # Mark utterance boundaries for the server (helps it pad/idle).
            if self._ws:
                tag = "speech_start" if isinstance(frame, TTSStartedFrame) else "speech_end"
                try:
                    await self._ws.send(json.dumps({"type": tag}))
                except Exception:  # noqa: BLE001
                    pass
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)
