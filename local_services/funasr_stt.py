"""FunASR (Paraformer) Mandarin STT as a Pipecat service.

FunASR runs in-process (no separate server) — it loads the Paraformer model on
the GPU and transcribes a complete, VAD-bounded utterance. We subclass
SegmentedSTTService so Pipecat hands us one utterance at a time (the upstream
Silero VAD decides the boundaries), which is the simplest reliable path for
Mandarin and avoids partial-decoding instability.

Model: `paraformer-zh` (offline, high accuracy). Add `punc` for punctuation.
First call downloads the checkpoint to ~/.cache/modelscope.

Requires: `pip install funasr modelscope torch`  (CUDA build of torch).
"""
from __future__ import annotations

import asyncio
from typing import AsyncGenerator

from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601


class FunASRSTTService(SegmentedSTTService):
    def __init__(self, *, language: str = "zh", sample_rate: int = 16000, **kwargs):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._language = language
        self._model = None  # lazy-loaded on first audio to keep startup fast

    def _ensure_model(self):
        if self._model is None:
            from funasr import AutoModel

            logger.info("Loading FunASR paraformer-zh (first run downloads weights)…")
            self._model = AutoModel(
                model="paraformer-zh",
                vad_model="fsmn-vad",
                punc_model="ct-punc",
                device="cuda",
                disable_update=True,
            )
            logger.info("FunASR model ready.")

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe one utterance (16-bit PCM mono) -> TranscriptionFrame."""
        import numpy as np

        try:
            self._ensure_model()
            await self.start_processing_metrics()
            pcm = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            # FunASR is CPU/GPU-bound and sync; run off the event loop.
            result = await asyncio.to_thread(
                self._model.generate, input=pcm, batch_size_s=300
            )
            await self.stop_processing_metrics()

            text = (result[0].get("text", "") if result else "").strip()
            if text:
                logger.debug(f"FunASR: {text}")
                yield TranscriptionFrame(
                    text, "", time_now_iso8601(), self._language
                )
        except Exception as e:  # noqa: BLE001
            logger.exception("FunASR STT failed")
            yield ErrorFrame(f"FunASR STT error: {e}")
