"""MicGate: ignore the user's mic until the avatar is ready.

Simli (cloud avatar) takes ~10s to warm its session when a client connects. If
the user talks during that window, their first message gets stuck behind the
cold start. This gate sits right after transport.input() and drops user audio
frames until the greeting finishes (BotStoppedSpeakingFrame, which the output
transport also pushes upstream) — so effectively the mic is "off" until the
system is ready. A safety timer opens it anyway if the greeting never completes.
"""
from __future__ import annotations

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# User-input frames we suppress while the system is warming up.
_BLOCKED = (InputAudioRawFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame)


class MicGate(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._ready = False

    def open(self):
        if not self._ready:
            self._ready = True
            logger.info("MicGate: avatar ready — mic enabled.")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # The avatar STARTING to speak (the greeting) = it's warm and on screen.
        # Open here so the mic goes live the same moment the avatar appears,
        # instead of waiting for the greeting to finish.
        if isinstance(frame, BotStartedSpeakingFrame):
            self.open()

        # While not ready, swallow the user's mic input so it can't queue up
        # behind the avatar's cold start.
        if (
            not self._ready
            and direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, _BLOCKED)
        ):
            return

        await self.push_frame(frame, direction)
