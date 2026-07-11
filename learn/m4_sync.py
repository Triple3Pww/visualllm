"""Module 4 toy -- Pixels, GPUs, and the sync contract.

The avatar renders at TARGET_FPS on paper. Under GPU contention it actually manages
REAL_FPS. The audio does not care -- it plays at exactly one second per second.

So: what do you do with the video?

    python learn/m4_sync.py live      # audio-master: ship audio now, video lags
    python learn/m4_sync.py steady    # video-master: pace audio to the real frames
    python learn/m4_sync.py steady --contention   # now the GPU gets busy mid-turn
    python learn/m4_sync.py steady --trt          # the renderer finally KEEPS UP

Four blanks are yours -- most of both release loops. By now you can write them.
"""
import argparse
import time

TARGET_FPS = 14.0     # what MUSETALK_FPS claims
SLOW_FPS = 10.0       # PyTorch render, sharing the GPU with CosyVoice: it cannot keep up
TRT_FPS = 16.0        # TensorRT render (MUSETALK_TRT=1): finally faster than the target
AUDIO_SECONDS = 8.0   # length of the turn's voice

FRAME_BUDGET = 1.0 / TARGET_FPS


def render_frames(seconds, real_fps, contention_at=None):
    """Yield (frame_index, wall_clock_when_it_finished_rendering).

    The renderer is simulated, not real -- but its speed relative to TARGET_FPS is the
    only property that matters. This function is given; do not change it.
    """
    n = int(seconds * TARGET_FPS)
    cost0 = 1.0 / real_fps
    t = 0.0
    for i in range(n):
        cost = cost0
        if contention_at is not None and t >= contention_at:
            cost = cost0 * 1.6      # CosyVoice starts synthesising. we get slower.
        t += cost
        yield i, t


def play_live(frames):
    """AUDIO-MASTER. The voice plays immediately, at real time. Video arrives when it can.

    Report the DRIFT: how far behind the audio each frame lands.
    """
    drifts = []
    for i, rendered_at in frames:
        # TODO(you) #1
        #   `audio_time` is when this frame's audio SHOULD be heard: i / TARGET_FPS.
        #   `drift` is how late the frame is: rendered_at - audio_time.
        #   Append drift to `drifts`.
        raise NotImplementedError("TODO(you) #1")
    return drifts


def play_steady(frames):
    """VIDEO-MASTER. Audio is released paced to the frames actually rendered.

    The voice WAITS for the renderer, so audio and video cannot drift. The cost is
    that a long stall makes the voice pause.
    """
    released = []
    for i, rendered_at in frames:
        # TODO(you) #2
        #   The audio for frame i may only be released once frame i EXISTS.
        #   So its release time is simply `rendered_at`. Append it to `released`.
        raise NotImplementedError("TODO(you) #2")

    # TODO(you) #3
    #   Drift is now zero by construction -- frame i is pinned to audio i. Instead the
    #   cost shows up as PAUSES. For each consecutive pair in `released`, the gap is
    #   released[k] - released[k-1]. A gap LONGER than FRAME_BUDGET means the voice
    #   had to wait. Return the list of those gaps.
    raise NotImplementedError("TODO(you) #3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["live", "steady"])
    ap.add_argument("--contention", action="store_true",
                    help="the GPU gets busy 3s in (CosyVoice starts talking too)")
    ap.add_argument("--trt", action="store_true",
                    help="TensorRT render: now the renderer is FASTER than the target")
    args = ap.parse_args()

    real_fps = TRT_FPS if args.trt else SLOW_FPS
    at = 3.0 if args.contention else None
    frames = render_frames(AUDIO_SECONDS, real_fps, contention_at=at)

    print("target fps : %.0f   (MUSETALK_FPS)" % TARGET_FPS)
    print("real fps   : %.0f   (%s)" % (
        real_fps, "TensorRT -- it keeps up" if args.trt else "PyTorch on a shared GPU -- it does not keep up"))
    print("turn length: %.0fs\n" % AUDIO_SECONDS)

    if args.mode == "live":
        drifts = play_live(frames)
        print("final drift: %.2fs   <-- the lips are this far behind the voice" % drifts[-1])
        print("max drift  : %.2fs" % max(drifts))
        # TODO(you) #4
        #   Print ONE sentence, in your own words, explaining why the drift GROWS with
        #   turn length instead of staying constant. (If you cannot, re-read the numbers
        #   above until you can. This is the module's real exam.)
        raise NotImplementedError("TODO(you) #4 -- write the sentence")
    else:
        gaps = play_steady(frames)
        stalls = [g for g in gaps if g > FRAME_BUDGET * 1.05]
        print("frame budget : %.3fs  (1 / target fps)" % FRAME_BUDGET)
        print("audio gaps   : %d of %d exceeded it" % (len(stalls), len(gaps)))
        print("worst gap    : %.3fs" % (max(gaps) if gaps else 0))
        print("\nDrift is ZERO -- frame i is pinned to audio i, by construction.")
        print("The renderer's slowness did not desync anything; it slowed the VOICE.")
        if not args.trt:
            print("\nBut look at how MANY gaps exceeded the budget. When the renderer is")
            print("ALWAYS below target, `steady` does not pause occasionally -- it stretches")
            print("the ENTIRE voice. Steady alone is not enough. It needs render headroom.")
            print("\nSo go get some:   python learn/m4_sync.py steady --trt")
        else:
            print("\nNow the renderer beats the target, so almost nothing waits. THAT is why")
            print("MUSETALK_TRT=1 is load-bearing and not a nice-to-have (P16): TensorRT did")
            print("not fix the sync logic -- the sync logic was right all along. It bought")
            print("back the HEADROOM the sync logic needed in order to cost nothing.")
        print("\nSee MUSETALK_SYNC_MODE in CLAUDE.md, and P16 for the real story.")


if __name__ == "__main__":
    main()
