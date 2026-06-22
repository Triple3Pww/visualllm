"""Transport-level proof of the A/V-sync root cause + fix (no browser needed).

Bug: in pipecat 1.3.0, a sync_with_audio OutputImageRawFrame is routed into the AUDIO
queue and rendered via `_video_images`, which `_video_task_handler` ONLY reads when
video_out_is_live is FALSE. With is_live=True (the old Ditto setting) the live handler
reads only `_video_queue`, so every sync_with_audio (lip-synced) frame is silently
dropped -> the video plays on a free-running clock and drifts vs the voice.

This drives pipecat's real MediaSender with a fake transport that records what actually
gets DRAWN, and asserts: is_live=True drops the synced turn frame; is_live=False draws it
(and still draws idle frames -> no freeze). Run: python -m scripts._sync_routing_test
"""
import asyncio

from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.frames.frames import OutputImageRawFrame, OutputAudioRawFrame

MediaSender = BaseOutputTransport.MediaSender
IMG = b"\x00" * (8 * 8 * 3)


class FakeTransport:
    def __init__(self):
        self.drawn = []  # frames that actually reached the wire (write_video_frame)

    def create_task(self, coro):
        coro.close()  # don't run the infinite task loops in this unit test
        return None

    def get_event_loop(self):
        return asyncio.get_running_loop()

    async def write_video_frame(self, frame):
        self.drawn.append(frame)

    async def write_audio_frame(self, frame):
        return True

    async def push_frame(self, *a, **k):
        pass


def _img(tag):
    f = OutputImageRawFrame(image=IMG, size=(8, 8), format="RGB")
    f.sync_with_audio = (tag == "turn")  # turn frames are tagged; idle frames are not
    f._tag = tag
    return f


async def _make(is_live):
    p = TransportParams(
        audio_out_enabled=True, video_out_enabled=True,
        video_out_is_live=is_live, video_out_width=8, video_out_height=8,
        video_out_framerate=12,
    )
    t = FakeTransport()
    s = MediaSender(t, destination=None, sample_rate=16000, audio_chunk_size=640, params=p)
    s._audio_queue = asyncio.Queue()
    s._video_queue = asyncio.Queue()
    return t, s


async def _drain_audio_queue(s):
    """Mimic _audio_task_handler: an image frame updates _video_images (the displayed
    source); audio frames are written. (base_output.py _handle_frame lines 738-743.)"""
    while not s._audio_queue.empty():
        frame = s._audio_queue.get_nowait()
        if isinstance(frame, OutputImageRawFrame):
            await s._set_video_image(frame)
        elif isinstance(frame, OutputAudioRawFrame):
            await s._transport.write_audio_frame(frame)


async def _draw_one_video_frame(s):
    """Mimic ONE iteration of _video_task_handler for the configured mode."""
    if s._params.video_out_is_live:
        if not s._video_queue.empty():
            img = s._video_queue.get_nowait()
            await s._draw_image(img)
    elif s._video_images:
        import itertools
        img = next(s._video_images)
        await s._draw_image(img)


async def scenario(is_live):
    t, s = await _make(is_live)
    # 1) idle frame arrives (untagged) -> route it, then the video task draws (idle period)
    await s.handle_image_frame(_img("idle"))
    await _draw_one_video_frame(s)
    # 2) a lip-synced turn frame arrives (tagged) -> route it, the audio task drains it,
    #    then the video task draws (during the turn)
    await s.handle_image_frame(_img("turn"))
    await _drain_audio_queue(s)
    await _draw_one_video_frame(s)
    return [getattr(f, "_tag", "?") for f in t.drawn]


async def main():
    live = await scenario(is_live=True)
    nonlive = await scenario(is_live=False)
    print(f"is_live=True  (OLD): drawn frames = {live}")
    print(f"is_live=False (NEW): drawn frames = {nonlive}")
    bug = "turn" not in live          # synced frame dropped under is_live -> the bug
    fixed = "turn" in nonlive          # synced frame drawn under non-live -> the fix
    idle_ok = "idle" in nonlive        # idle frame still drawn -> no freeze
    print(f"\nBUG reproduced (synced frame dropped @ is_live=True): {bug}")
    print(f"FIX (synced frame drawn @ is_live=False):            {fixed}")
    print(f"idle still drawn @ is_live=False (no freeze):         {idle_ok}")
    ok = bug and fixed and idle_ok
    print("\nRESULT:", "PASS - root cause + fix confirmed" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
