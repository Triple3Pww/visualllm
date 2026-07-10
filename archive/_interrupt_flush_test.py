"""Offline proof of the barge-in FLUSH fix (no server/GPU/WebRTC needed).

Bug: on an InterruptionFrame the client drained the audio queue + told the server to reset,
but frames the server had ALREADY rendered kept arriving on the ws. They hit `_on_frame` with
`_video_active=False` and were forwarded as IDLE animation -> the avatar kept lip-moving with
NO voice, and stragglers after the next `video_start` bled into the new turn.

Fix (client side, this test): a `_flushing` flag set by the InterruptionFrame handler, cleared
by the next TTSStartedFrame handler; `_on_frame` DROPS every frame while flushing. This drives
the REAL process_frame handlers (not hand-set flags) so it guards the wiring too, then checks
`_on_frame`'s drop across before / during / after the flush.

Run: python -m archive._interrupt_flush_test
(no torch/onnx import -- only websockets + the pipecat frame types + a bare TaskManager)
"""
import asyncio
import os

os.environ.setdefault("MUSETALK_SYNC_MODE", "steady")  # video-master (the mode that leaked)

from local_services.musetalk_video import MuseTalkVideoService
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import InterruptionFrame, OutputImageRawFrame, TTSStartedFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

SIZE = (8, 8)
IMG = b"\x01" * (SIZE[0] * SIZE[1] * 3)
DOWN = FrameDirection.DOWNSTREAM


async def main():
    tm = TaskManager()
    tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
    svc = MuseTalkVideoService(base_url="http://localhost:8002", fps=12, image_size=SIZE)
    await svc.setup(FrameProcessorSetup(
        clock=SystemClock(), task_manager=tm, pipeline_worker=None))

    imgs = []   # only the rendered frames that reached the transport (ignore control frames)

    async def _record(frame, direction=DOWN):
        if isinstance(frame, OutputImageRawFrame):
            imgs.append(frame)

    svc.push_frame = _record   # intercept everything that would reach the transport

    # Baseline: between turns, a server frame is forwarded as idle animation.
    svc._video_active = False
    await svc._on_frame(IMG)
    assert not svc._flushing, "should not be flushing at baseline"
    assert len(imgs) == 1, f"idle frame should forward: got {len(imgs)}"

    # Barge-in: the REAL InterruptionFrame handler must arm the flush.
    await svc.process_frame(InterruptionFrame(), DOWN)
    assert svc._flushing, "InterruptionFrame handler must set _flushing=True"

    # In-flight old-turn frames still draining from the server -> MUST be dropped.
    for _ in range(5):
        await svc._on_frame(IMG)
    assert len(imgs) == 1, (
        f"flushing must drop every in-flight frame: forwarded grew to {len(imgs)}")

    # New turn: the REAL TTSStartedFrame handler must end the flush.
    await svc.process_frame(TTSStartedFrame(), DOWN)
    assert not svc._flushing, "TTSStartedFrame handler must clear _flushing"

    # Flush over -> frames forward again.
    svc._video_active = False
    await svc._on_frame(IMG)
    assert len(imgs) == 2, f"post-flush frame should forward: got {len(imgs)}"

    print("PASS: interrupt arms flush -> in-flight frames dropped -> next turn clears flush")


if __name__ == "__main__":
    asyncio.run(main())
