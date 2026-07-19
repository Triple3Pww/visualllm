"""Offline SenseVoice STT that DRIVES ITS OWN TURN-TAKING (self-segmenting).

Why not SegmentedSTTService: that base waits for VADUserStopped frames from the transport,
and this pipeline's transport never pushes them to the STT (turn-taking runs through Smart Turn
downstream). So an offline STT gets audio but no "user stopped" cue and never fires. Verified
live 2026-07-18: audio arrived, speaking stayed False forever, the buffer never grew.

The fix mirrors sherpa_stt.py's proven design: a lightweight streaming zipformer runs ONLY as the
endpoint detector (it decides start/stop of speech and is robust to a quiet mic on this box), and
at each endpoint the buffered utterance is transcribed by SenseVoice-Small -- the accurate,
noise-robust model (A/B 2026-07-18: held at 5dB SNR where the zipformer garbled). The zipformer's
own text is used only for the fast interim bubble; the FINAL transcript is SenseVoice's.

So this service emits the SAME frames sherpa did (VADUserStarted on onset, a final TranscriptionFrame
+ VADUserStopped at the endpoint), which is what actually drove turns here -- only the final text is
upgraded to SenseVoice. zh output -> Traditional (zh-TW) via OpenCC s2twp, as elsewhere.
"""
from __future__ import annotations

import asyncio
import os
import time
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


# Process-wide model cache. The pipeline is rebuilt on EVERY WebRTC connect (pipecat's dev
# runner calls the bot entrypoint per connection), so SenseVoiceSTTService.__init__ runs per
# connect -- and loading these sherpa-onnx models from disk costs ~1.6-2.7s, paid on every
# single Connect (measured 2026-07-18). The recognizers are stateless model holders: all
# per-session state lives in streams (created per instance), so one cached set is safely
# reused across sequential single-client sessions. Keyed by everything that shapes the model.
_MODEL_CACHE: dict = {}


def _load_recognizers(endpoint_model_dir: str, model_dir: str, provider: str,
                      endpoint_silence: float):
    """Build (or return the cached) endpoint zipformer + SenseVoice recognizers."""
    key = (endpoint_model_dir, model_dir, provider, round(endpoint_silence, 3))
    hit = _MODEL_CACHE.get(key)
    if hit is not None:
        logger.info("SenseVoice STT: reusing cached models (skipped disk load).")
        return hit
    import sherpa_onnx

    # Endpoint detector = the streaming zipformer (CPU, ~0 VRAM), used ONLY to decide when
    # speech starts/ends. Same config as sherpa_stt.py (proven robust to a quiet mic here).
    ep = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=_find(endpoint_model_dir, "tokens.txt"),
        encoder=_find(endpoint_model_dir, "encoder-epoch-99-avg-1.int8.onnx", "encoder-epoch-99-avg-1.onnx"),
        decoder=_find(endpoint_model_dir, "decoder-epoch-99-avg-1.onnx", "decoder-epoch-99-avg-1.int8.onnx"),
        joiner=_find(endpoint_model_dir, "joiner-epoch-99-avg-1.int8.onnx", "joiner-epoch-99-avg-1.onnx"),
        num_threads=2, decoding_method="greedy_search", enable_endpoint_detection=True,
        rule1_min_trailing_silence=max(endpoint_silence, 1.2),
        rule2_min_trailing_silence=endpoint_silence, rule3_min_utterance_length=300)

    # Transcriber = SenseVoice-Small offline (use_itn=1 -> punctuation). provider="cuda" falls
    # back to CPU on the pip wheel (still real-time, RTF ~0.016, 0 GPU) -- fine for 8GB deploy.
    sv = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=_find(model_dir, "model.int8.onnx"), tokens=_find(model_dir, "tokens.txt"),
        num_threads=2, use_itn=True, provider=provider)

    _MODEL_CACHE[key] = (ep, sv)
    return ep, sv


_OPENCC = None


def _get_opencc():
    """Cached OpenCC s2twp converter (zh -> Traditional zh-TW), shared across connects."""
    global _OPENCC
    if _OPENCC is None:
        import opencc
        _OPENCC = opencc.OpenCC("s2twp")
    return _OPENCC


class SenseVoiceSTTService(STTService):
    def __init__(self, *, model_dir: str, endpoint_model_dir: str, provider: str = "cuda",
                 to_traditional: bool = True, endpoint_silence: float = 0.5,
                 pause_while_bot_speaks: bool = False, sample_rate: int | None = None, **kwargs):
        kwargs.setdefault(
            "settings", STTSettings(model="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8",
                                    language=None, extra={}))
        # ttfs_p99_latency = how long the turn strategy waits AFTER deciding you're done (Smart
        # Turn COMPLETE) for the STT's final transcript before firing the LLM. This service emits
        # its final synchronously with its own endpoint (the transcript is already in hand at the
        # turn boundary), so its real ttfs is ~0. The base default (1.0s, a cloud-STT guess) added
        # up to ~1s of PRE-t0 felt latency every turn -- invisible to TTFO (P54). 0.1s trims it.
        kwargs.setdefault("ttfs_p99_latency", 0.1)
        super().__init__(sample_rate=sample_rate, **kwargs)

        # Cached across connects (see _load_recognizers); the stream is per-instance state.
        self._ep, self._sv = _load_recognizers(
            endpoint_model_dir, model_dir, provider, endpoint_silence)
        self._stream = self._ep.create_stream()

        self._cc = None
        if to_traditional:
            self._cc = _get_opencc()

        self._buf = bytearray()
        # Pre-roll kept while NOT speaking so the utterance's first syllable isn't clipped. The
        # zipformer flags onset a beat AFTER speech truly starts, so a short pre-roll drops the
        # opening of a quick word (e.g. "你好" -> SenseVoice only saw "好"). 1.0s covers the onset
        # lag with margin; leading silence is harmless to SenseVoice. (matches pipecat's own 1s.)
        self._preroll_bytes = int(1.0 * self.sample_rate) * 2 if self.sample_rate else 32000
        self._speaking = False
        self._last_partial = ""
        # Live-preview (interim) throttle: run SenseVoice on the growing buffer at most this often
        # so the bubble previews in SenseVoice quality (not the rough zipformer) without flooding
        # the CPU. The zipformer partial is used only as the "new speech happened" trigger.
        self._last_interim_t = 0.0
        self._interim_min_interval = 0.4
        # Continuation merge is NOT done here: each endpoint emits a clean segment and the LLM
        # aggregator concatenates segments within a turn for free. Stitching text in this service
        # duplicated when the aggregator also stacked (measured 2026-07-18), so it was removed.
        self._pause_while_bot_speaks = pause_while_bot_speaks
        self._bot_speaking = False
        logger.info(f"SenseVoice STT ready (transcriber={model_dir}, endpoint={endpoint_model_dir}, "
                    f"provider={provider}, traditional={to_traditional})")

    def _conv(self, text: str) -> str:
        return self._cc.convert(text) if self._cc else text

    def _endpoint(self, audio: bytes) -> tuple[str, bool]:
        """Feed one chunk to the zipformer; return (streaming partial, is_endpoint). CPU, in a
        thread (sherpa-onnx releases the GIL) so it never stalls the pipeline loop."""
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        self._stream.accept_waveform(self.sample_rate, samples)
        while self._ep.is_ready(self._stream):
            self._ep.decode_stream(self._stream)
        return self._ep.get_result(self._stream).strip(), self._ep.is_endpoint(self._stream)

    def _transcribe(self, pcm: bytes) -> str:
        """Accurate final transcript of the whole buffered utterance via SenseVoice."""
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            return ""
        s = self._sv.create_stream()
        s.accept_waveform(self.sample_rate, samples)
        self._sv.decode_stream(s)
        return self._conv(s.result.text.strip())

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        if self._pause_while_bot_speaks and self._bot_speaking:
            return
        loop = asyncio.get_running_loop()
        now = time.monotonic()

        # Accumulate audio for the CURRENT speech segment. The buffer is CLEARED on each endpoint,
        # so each segment is transcribed independently and cleanly -- no overlap, no cumulative
        # re-transcription (that was what duplicated the text). While idle, keep only a short
        # pre-roll so a fresh utterance's first syllable isn't clipped by the detector's onset lag.
        self._buf += audio
        if not self._speaking and len(self._buf) > self._preroll_bytes:
            del self._buf[:-self._preroll_bytes]

        partial, is_endpoint = await loop.run_in_executor(None, self._endpoint, audio)

        if partial and not self._speaking:
            self._speaking = True
            yield VADUserStartedSpeakingFrame()

        if is_endpoint:
            if self._speaking:
                # Emit the clean segment transcript. When the eager endpoint fires mid-sentence and
                # the user keeps talking, each segment is a separate final -- the LLM aggregator
                # CONCATENATES them within the turn, so the merge is free and never duplicates (we
                # do NOT stitch text ourselves; that double-counted when the aggregator also stacked).
                text = await loop.run_in_executor(None, self._transcribe, bytes(self._buf))
                if text:
                    yield TranscriptionFrame(text, "", time_now_iso8601())
                yield VADUserStoppedSpeakingFrame()
                self._speaking = False
            self._buf.clear()             # clean, non-overlapping next segment
            self._ep.reset(self._stream)
            self._last_partial = ""
        elif partial and partial != self._last_partial and self._speaking:
            # Live preview in SenseVoice quality (not the rough zipformer text): re-run SenseVoice on
            # the current segment.
            self._last_partial = partial
            if now - self._last_interim_t >= self._interim_min_interval:
                self._last_interim_t = now
                itext = await loop.run_in_executor(None, self._transcribe, bytes(self._buf))
                if itext:
                    yield InterimTranscriptionFrame(itext, "", time_now_iso8601())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if not self._pause_while_bot_speaks:
            return
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            if self._bot_speaking:
                self._bot_speaking = False
                self._buf.clear()
                self._ep.reset(self._stream)
                self._speaking = False
                self._last_partial = ""
