"""StageTap — a pass-through FrameProcessor that observes the pipeline.

One tap is inserted just AFTER each real stage. It records nothing about the
frames' contents and never alters or drops them — it forwards every frame
unchanged (the exact pattern TtfoMeter already uses), then reports a *semantic*
event to the StatusBus ("stt produced output", "user stopped", "error here").

This file owns the version-sensitive Pipecat frame imports (like the stage
factories do); the bus stays Pipecat-free. If a frame path drifts, it drifts
here, in isolation, and preflight catches it.
"""
from __future__ import annotations

# Version-sensitive imports kept together (see pipeline/stages/*.py).
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    LLMTextFrame,
    OutputImageRawFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from pipeline.debug.status_bus import STAGE_KEYS, bus

# Which frame type, flowing downstream out of a stage, counts as that stage
# having "produced output" (and therefore being alive). Earliest signal wins,
# so STT uses interim transcriptions too.
_OUTPUT_TYPES: dict[str, tuple[type, ...]] = {
    "vad": (UserStartedSpeakingFrame, UserStoppedSpeakingFrame),
    "stt": (TranscriptionFrame, InterimTranscriptionFrame),
    "llm": (LLMTextFrame,),
    "tts": (TTSAudioRawFrame,),
    "avatar": (OutputImageRawFrame,),
    "out": (),  # judged by BotStartedSpeakingFrame, handled specially below
}


def _err_msg(frame: ErrorFrame) -> str:
    msg = getattr(frame, "error", None) or str(frame)
    return str(msg)[:200]


class StageTap(FrameProcessor):
    """Observe-and-forward. `stage` is one of the StatusBus stage keys."""

    def __init__(self, stage: str, **kwargs):
        super().__init__(**kwargs)
        self._stage = stage
        self._output_types = _OUTPUT_TYPES.get(stage, ())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        # Observation must never break the pipeline — swallow any bookkeeping error.
        try:
            self._observe(frame, direction)
        except Exception:  # noqa: BLE001 — a debug tap can't be allowed to throw
            pass
        await self.push_frame(frame, direction)

    def _observe(self, frame: Frame, direction: FrameDirection) -> None:
        # Turn boundaries are global (idempotent in the bus), so any tap may report
        # them — whichever one the frame passes through first.
        if isinstance(frame, UserStoppedSpeakingFrame):
            bus.on_user_stopped()
        if isinstance(frame, BotStartedSpeakingFrame):
            bus.on_bot_started()
            bus.mark_output("out")  # the bot is now speaking -> 'out' is live

        if isinstance(frame, ErrorFrame):
            bus.mark_error(self._stage, _err_msg(frame))

        if (self._output_types
                and direction == FrameDirection.DOWNSTREAM
                and isinstance(frame, self._output_types)):
            bus.mark_output(self._stage)


def make_taps() -> dict[str, StageTap]:
    """Build one tap per stage. main.py interleaves these into the stage list."""
    return {key: StageTap(key) for key in STAGE_KEYS}
