"""Deterministic proof of the P11 root cause + fix (no server/GPU/browser needed).

BUG (P11 mechanism, steady sync): MuseTalkVideoService forwards TTSStoppedFrame downstream
IMMEDIATELY while the turn's voice is still HELD in _abuf (released later, paced to rendered
frames). pipecat's output transport fires BotStoppedSpeaking on the TTSStoppedFrame, then the
held audio released afterwards re-fires BotStartedSpeaking -- and with the screech fix's
BOT_VAD_STOP_FALLBACK_SECS=600 nothing ever fires BotStopped again. Echo-guard's mic mute
keys on BotStopped -> mic STUCK MUTED after the first turn (docs/PROBLEMS-AND-FIXES.md P11).

FIX: hold the TTSStoppedFrame in the avatar client and release it from _drain_audio, AFTER
the turn's voice has fully gone downstream -- so the transport sees [audio..., TTSStopped]
in true playback order and BotStopped fires exactly once, at real end of speech.

Asserts, driving the REAL MuseTalkVideoService in steady mode with a fake downstream:
  1. normal turn:      TTSStopped reaches downstream AFTER the last released audio frame
  2. interrupted turn: a held TTSStopped is DROPPED (the transport emits its own BotStopped
                       on InterruptionFrame; forwarding ours later would mid-turn-unmute)
  3. dead-server turn: video_end never arrives -> the next TTSStarted flushes the held stop
                       BEFORE the new turn (the previous turn still closes; no stuck state)

Run:  python archive/_tts_stop_order_test.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("MUSETALK_SYNC_MODE", "steady")

from pipecat.frames.frames import (  # noqa: E402
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection  # noqa: E402

from local_services.musetalk_video import (  # noqa: E402
    FRAME_HDR,
    FRAME_MAGIC,
    MUSETALK_SR,
    MuseTalkVideoService,
)

SIDE = 8
RGB = b"\x7f" * (SIDE * SIDE * 3)
DOWN = FrameDirection.DOWNSTREAM


async def _noop(*_a, **_k):
    return None


def _make():
    svc = MuseTalkVideoService(base_url="http://fake:8002", fps=14, image_size=(SIDE, SIDE))
    pushed = []

    async def _record(frame, direction=DOWN):
        pushed.append(frame)

    svc.push_frame = _record
    # Bare instance: no task manager / metrics behind the base class's interruption path.
    svc._start_interruption = _noop
    svc.stop_all_metrics = _noop
    return svc, pushed


def _audio(ms=20):
    n = MUSETALK_SR * ms // 1000
    return TTSAudioRawFrame(audio=b"\x01\x00" * n, sample_rate=MUSETALK_SR, num_channels=1)


async def _feed_turn(svc, n_audio=3):
    await svc.process_frame(TTSStartedFrame(), DOWN)
    total = 0
    for _ in range(n_audio):
        f = _audio()
        total += len(f.audio) // 2
        await svc.process_frame(f, DOWN)
    await svc.process_frame(TTSStoppedFrame(), DOWN)
    return total


def _order(pushed):
    """(index of last released audio frame, index of TTSStoppedFrame) in the downstream record."""
    last_audio = max((i for i, f in enumerate(pushed) if isinstance(f, TTSAudioRawFrame)), default=-1)
    stop = next((i for i, f in enumerate(pushed) if isinstance(f, TTSStoppedFrame)), -1)
    return last_audio, stop


async def scenario_normal():
    """Full turn with server markers: stop must land after the last audio."""
    svc, pushed = _make()
    total = await _feed_turn(svc)
    await svc._on_marker({"type": "proto", "v": 2})
    await svc._on_marker({"type": "video_start"})
    # One real rendered frame covering ALL the turn's audio -> _advance releases everything.
    await svc._on_frame(FRAME_HDR.pack(FRAME_MAGIC, 0, total) + RGB)
    await svc._on_marker({"type": "video_end"})
    svc._cancel_fallback()
    last_audio, stop = _order(pushed)
    released = sum(isinstance(f, TTSAudioRawFrame) for f in pushed)
    ok = released == 3 and stop != -1 and stop > last_audio
    print(f"normal turn:      released_audio={released}/3 last_audio@{last_audio} "
          f"stop@{stop} -> {'PASS' if ok else 'FAIL (stop before audio = P11)'}")
    return ok


async def scenario_interrupt():
    """Barge-in with a held stop: it must be dropped, not forwarded."""
    svc, pushed = _make()
    await _feed_turn(svc)
    await svc.process_frame(InterruptionFrame(), DOWN)
    svc._cancel_fallback()
    _last_audio, stop = _order(pushed)
    held = getattr(svc, "_held_tts_stop", None)
    ok = stop == -1 and held is None
    print(f"interrupted turn: stop_forwarded={stop != -1} held_left={held is not None} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


async def scenario_dead_server():
    """video_end never arrives: the NEXT TTSStarted must flush the held stop first."""
    svc, pushed = _make()
    await _feed_turn(svc)                      # turn 1: no server markers ever
    await svc.process_frame(TTSStartedFrame(), DOWN)   # turn 2 begins
    svc._cancel_fallback()
    stop = next((i for i, f in enumerate(pushed) if isinstance(f, TTSStoppedFrame)), -1)
    start2 = max(i for i, f in enumerate(pushed) if isinstance(f, TTSStartedFrame))
    ok = stop != -1 and stop < start2
    print(f"dead-server turn: stop@{stop} next_start@{start2} "
          f"-> {'PASS' if ok else 'FAIL (previous turn never closes)'}")
    return ok


async def main():
    r1 = await scenario_normal()
    r2 = await scenario_interrupt()
    r3 = await scenario_dead_server()
    ok = r1 and r2 and r3
    print("\nRESULT:", "PASS - TTSStopped reaches the transport at true end of speech"
          if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
