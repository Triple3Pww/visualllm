"""EmotionTagger -- the server side of "let the AI move the model".

CHARACTER_MODE asks the LLM to begin each reply with one emotion tag, e.g.
`[happy] ดีใจจังเลยค่ะ`. This processor sits between the LLM and the TTS and:
  1. captures that leading `[emotion]` tag from the streamed text,
  2. pushes it to the client out-of-band as an `RTVIServerMessageFrame` (serialized
     by the runner's existing RTVI layer -> the client's onServerMessage), where it
     drives the VRM facial expression,
  3. forwards the **tag-stripped** text to the TTS, so the tag is never spoken.

We push the RTVI frame ourselves rather than adding our own RTVIProcessor — a second
processor breaks the prebuilt runner's bot-ready handshake. The client also infers
emotion from the spoken text as a fallback, so this is best-effort.

Only the short leading tag is buffered, so first-audio latency is barely affected.
A reply with no leading tag passes straight through unchanged.
"""
from __future__ import annotations

import re

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame

EMOTIONS = {"neutral", "happy", "angry", "sad", "relaxed", "surprised"}
_TAG_RE = re.compile(r"^\s*\[([a-zA-Z]+)\]\s*")
# Give up looking for a leading tag once this much text has streamed without one.
_MAX_BUFFER = 24


class EmotionTagger(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._buf = ""
        self._capturing = False
        self._resolved = False

    async def _emit_emotion(self, tag: str, direction: FrameDirection):
        # best-effort: the runner's RTVI layer serializes this to onServerMessage
        try:
            await self.push_frame(
                RTVIServerMessageFrame(data={"type": "emotion", "value": tag}),
                direction,
            )
        except Exception:  # noqa: BLE001 -- emotion is decorative, never block speech
            pass

    async def _flush(self, direction: FrameDirection):
        if self._buf:
            await self.push_frame(LLMTextFrame(text=self._buf), direction)
            self._buf = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            # new assistant turn -> start hunting for a leading tag
            self._buf = ""
            self._capturing = True
            self._resolved = False
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMTextFrame) and self._capturing and not self._resolved:
            self._buf += frame.text
            m = _TAG_RE.match(self._buf)
            if m:
                tag = m.group(1).lower()
                rest = self._buf[m.end():]
                self._buf = ""
                self._resolved = True
                self._capturing = False
                if tag in EMOTIONS:
                    await self._emit_emotion(tag, direction)
                if rest:
                    await self.push_frame(LLMTextFrame(text=rest), direction)
                return
            # no tag yet -- if it's clearly not a leading tag, stop buffering and flush
            if len(self._buf) >= _MAX_BUFFER or ("]" in self._buf and not self._buf.lstrip().startswith("[")):
                self._resolved = True
                self._capturing = False
                await self._flush(direction)
            return  # keep buffering the short head otherwise

        if isinstance(frame, LLMFullResponseEndFrame):
            # safety: flush anything still buffered (very short replies)
            if self._capturing:
                await self._flush(direction)
            self._capturing = False
            self._resolved = False
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)
