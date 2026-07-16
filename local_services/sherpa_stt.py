"""Pipecat STREAMING STT using sherpa-onnx (in-process, CPU, ~0 VRAM).

Unlike the segmented SenseVoice service, this is a TRUE streaming STT: each audio chunk
is fed to a sherpa-onnx OnlineRecognizer with built-in endpoint detection, so it does
NOT depend on the transport's energy-VAD firing (which was unreliable on this box).

Crucially, this service DRIVES TURN-TAKING from the ASR endpoint detector: it emits
VADUserStartedSpeakingFrame when recognized speech begins and VADUserStoppedSpeakingFrame
(+ the final TranscriptionFrame) when sherpa detects end-of-utterance. The downstream
LLMUserAggregator flushes the user turn on VADUserStoppedSpeakingFrame, so this works even
when the energy-VAD never fires (e.g. quiet/attenuated mic audio). Endpoint detection runs
on the acoustic model output, not an energy threshold, so it is robust to low input level.

Bilingual zh-en; zh output is converted to Traditional (zh-TW) via OpenCC (s2twp).
Model: sherpa-onnx streaming zipformer bilingual (k2-fsa), int8, downloaded under models/.
"""
from __future__ import annotations

import asyncio
import os
from typing import AsyncGenerator

import numpy as np
from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import STTService
from pipecat.utils.time import time_now_iso8601


def _find(model_dir: str, *names: str) -> str:
    for n in names:
        p = os.path.join(model_dir, n)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"none of {names} found in {model_dir}")


class SherpaStreamingSTTService(STTService):
    def __init__(self, *, model_dir: str, to_traditional: bool = True,
                 endpoint_silence: float = 0.5, pause_while_bot_speaks: bool = False,
                 sample_rate: int | None = None, **kwargs):
        # Bilingual model, language auto-detected per utterance; declare model/language so
        # Pipecat's STTSettings.validate_complete doesn't log a (harmless) NOT_GIVEN error.
        kwargs.setdefault(
            "settings", STTSettings(model="sherpa-onnx-streaming-zipformer-zh-en", language=None, extra={})
        )
        # ttfs_p99_latency = post-speech-end delay until the FINAL transcript is ready. The turn
        # strategy (Smart Turn) waits this long after deciding the turn is over, to let a slow
        # streaming STT's final land -- pure pre-answer dead time. Sherpa yields its final
        # TranscriptionFrame *synchronously* with the endpoint/VADUserStopped (it is already in hand
        # at the turn boundary), so its real ttfs is ~0. The base default (1.0s, a cloud-STT guess)
        # was adding ~1s to every turn's pre-t0 wait -- measured in the mic-to-ear Capture segment.
        # 0.1s (non-zero, to avoid the stop_secs>=ttfs "collapse" path) trims that ~0.9s.
        kwargs.setdefault("ttfs_p99_latency", 0.1)
        super().__init__(sample_rate=sample_rate, **kwargs)
        import sherpa_onnx

        # int8 encoder/joiner = much smaller + faster on CPU at negligible accuracy cost.
        self._rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=_find(model_dir, "tokens.txt"),
            encoder=_find(model_dir, "encoder-epoch-99-avg-1.int8.onnx", "encoder-epoch-99-avg-1.onnx"),
            decoder=_find(model_dir, "decoder-epoch-99-avg-1.onnx", "decoder-epoch-99-avg-1.int8.onnx"),
            joiner=_find(model_dir, "joiner-epoch-99-avg-1.int8.onnx", "joiner-epoch-99-avg-1.onnx"),
            num_threads=2,
            decoding_method="greedy_search",
            enable_endpoint_detection=True,
            # rule2 = trailing silence (s) after speech that FIRES the query. The "fire easier"
            # knob (SHERPA_ENDPOINT_SILENCE). rule1 (no speech yet) tracks it so a brief utterance
            # still fires promptly; rule3 caps runaway length.
            rule1_min_trailing_silence=max(endpoint_silence, 1.2),
            rule2_min_trailing_silence=endpoint_silence,
            rule3_min_utterance_length=300,
        )
        self._stream = self._rec.create_stream()
        self._speaking = False
        # Pause decoding while the bot speaks (avoids transcribing the avatar's own voice as a
        # phantom turn) ONLY when echo-guard is on. Historical reason for the gate: under steady
        # sync BotStoppedSpeakingFrame never fired (P11) so the pause would get STUCK True after
        # the first bot turn and drop every later mic frame. That blocker is FIXED (P53,
        # 2026-07-15: the avatar client holds TTSStoppedFrame until the voice drains, so
        # BotStopped fires at true end of speech even under steady) -- but the gate stays,
        # matching echo-guard's own posture: default OFF (barge-in/headphones baseline), and
        # un-gating deserves its own live test, not a ride-along. docs/PROBLEMS-AND-FIXES.md P11/P53.
        self._pause_while_bot_speaks = pause_while_bot_speaks
        self._bot_speaking = False
        self._last_partial = ""
        self._cc = None
        if to_traditional:
            import opencc
            self._cc = opencc.OpenCC("s2twp")
        logger.info(f"Sherpa streaming STT ready (model={model_dir}, traditional={to_traditional})")

    def _conv(self, text: str) -> str:
        return self._cc.convert(text) if self._cc else text

    def _decode(self, audio: bytes) -> tuple[str, bool]:
        """Feed one chunk + decode whatever is ready, returning (partial text, is_endpoint).
        Pure CPU (the zipformer forward pass); runs in a thread so it never blocks the pipeline's
        asyncio loop -- on this single-loop box, decoding inline starved the real-time TTS->avatar
        audio pacing and pushed lip-start ~4s late (the decode is fast but fires on every mic frame,
        so it monopolised loop time slices). sherpa-onnx releases the GIL in its C++ core, so the
        loop runs free while this thread decodes. run_stt awaits this, so calls stay sequential ->
        no concurrent access to the (non-thread-safe) recognizer/stream."""
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        self._stream.accept_waveform(self.sample_rate, samples)
        while self._rec.is_ready(self._stream):
            self._rec.decode_stream(self._stream)
        return self._rec.get_result(self._stream).strip(), self._rec.is_endpoint(self._stream)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        # Called per audio frame (streaming). When echo-guard is on, PAUSE while the bot is
        # speaking so the avatar's own voice isn't transcribed as a phantom user turn. Default is
        # OFF (mic always live -- barge-in/headphones), because under steady sync the resume signal
        # never arrives and the pause would strand (P11); see __init__.
        if self._pause_while_bot_speaks and self._bot_speaking:
            return
        text, is_endpoint = await asyncio.get_running_loop().run_in_executor(
            None, self._decode, audio)

        # Speech onset: drive the user turn from the ASR (not the energy-VAD).
        if text and not self._speaking:
            self._speaking = True
            yield VADUserStartedSpeakingFrame()

        if is_endpoint:
            if text:
                yield TranscriptionFrame(self._conv(text), "", time_now_iso8601())
            if self._speaking:
                yield VADUserStoppedSpeakingFrame()
                self._speaking = False
            self._rec.reset(self._stream)
            self._last_partial = ""
        elif text and text != self._last_partial:
            self._last_partial = text
            yield InterimTranscriptionFrame(self._conv(text), "", time_now_iso8601())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Track when the bot is speaking so run_stt can pause (above), only relevant with
        # echo-guard on. Bot speaking frames are pushed UPSTREAM from the output transport, so they
        # reach this stage; super() forwards them.
        await super().process_frame(frame, direction)
        if not self._pause_while_bot_speaks:
            return
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            if self._bot_speaking:
                self._bot_speaking = False
                # Drop any stale partial captured around the bot's turn so the next user
                # utterance starts from a clean stream.
                self._rec.reset(self._stream)
                self._speaking = False
                self._last_partial = ""
