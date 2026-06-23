"""Proof of the "avatar finishes before the audio" root cause + fix (no GPU needed).

Bug: the MuseTalk server slices each render segment as `int(16000/fps) * SEG_FRAMES`
samples, but the renderer counts frames as `floor(len/sr * fps)` (MuseTalk's
get_whisper_chunk). When fps does NOT divide 16000 evenly (e.g. 12), `int(16000/fps)`
truncates, so an 8-frame segment lands at floor(7.998)=7 -- one lip frame lost PER
segment. Over a long reply the deficit accumulates (~12.5%) and the lips finish ~1-2s
before the voice. At fps=20/25 (16000 divisible) there was no loss, so this is a
regression introduced by lowering MUSETALK_FPS to 12.

This replicates the EXACT two formulas the server uses -- the segment sizing
(`samples_for_frames`) and the renderer's frame count (`floor(len/sr*fps)`) -- and
asserts every segment yields SEG_FRAMES, for the fps values that bite. No model/GPU.

Run: python -m archive._frame_deficit_repro_test
"""
import math

SR = 16000
SEG_FRAMES = 8


def rendered_frames(n_samples: int, fps: int) -> int:
    """How many frames MuseTalk's get_whisper_chunk emits for n_samples of audio.
    Mirrors audio_processor.py: num_frames = floor((librosa_length / sr) * fps)."""
    return math.floor((n_samples / SR) * fps)


def old_seg_samples(fps: int) -> int:
    """The OLD (buggy) sizing: int(sr/fps) per frame, times SEG_FRAMES."""
    return int(SR / fps) * SEG_FRAMES


def new_seg_samples(n_frames: int, fps: int) -> int:
    """The FIX: size to the frame's upper sample boundary so floor() can't drop a
    fractional frame. Matches MuseTalkEngine.samples_for_frames."""
    return math.ceil(n_frames * SR / fps)


def main() -> None:
    bite_fps = [12, 15, 18, 24, 9]      # fps that do NOT divide 16000 evenly
    safe_fps = [20, 25, 10, 16, 8]      # fps that DO -- never lost frames

    # 1. Demonstrate the bug exists on the current default (12).
    old = rendered_frames(old_seg_samples(12), 12)
    assert old == SEG_FRAMES - 1, f"expected the bug (7), got {old}"
    print(f"[repro] OLD sizing @12fps -> {old} frames/segment (want {SEG_FRAMES}) -- BUG confirmed")

    # 2. The fix yields exactly SEG_FRAMES for every fps, biting or not.
    for fps in bite_fps + safe_fps:
        got = rendered_frames(new_seg_samples(SEG_FRAMES, fps), fps)
        assert got == SEG_FRAMES, f"FIX failed @ {fps}fps: {got} != {SEG_FRAMES}"
    print(f"[fix] NEW sizing -> exactly {SEG_FRAMES} frames/segment for all fps {bite_fps + safe_fps}")

    # 3. End-to-end: a 13.5s reply at 12fps should render ~162 frames (audio*fps),
    #    not the ~141 the old path produced (which is what finished early).
    audio_s = 13.5
    total = int(audio_s * SR)
    # old path: full segments + a truncating tail
    seg_old = old_seg_samples(12)
    old_total, buf = 0, total
    while buf >= seg_old:
        old_total += rendered_frames(seg_old, 12)
        buf -= seg_old
    old_total += rendered_frames(buf, 12)  # tail (old: no ceil pad)
    # new path
    seg_new = new_seg_samples(SEG_FRAMES, 12)
    new_total, buf = 0, total
    while buf >= seg_new:
        new_total += rendered_frames(seg_new, 12)
        buf -= seg_new
    f_final = math.ceil(buf / SR * 12)
    new_total += rendered_frames(new_seg_samples(f_final, 12), 12) if buf > 0 else 0
    want = math.floor(audio_s * 12)
    print(f"[e2e] {audio_s}s reply @12fps: old={old_total} new={new_total} want~{want}")
    assert old_total < want - 10, "old path should be well short (it was ~141)"
    assert abs(new_total - want) <= 1, f"new path should match audio length, got {new_total}"
    print("OK: fix removes the per-segment frame deficit.")


if __name__ == "__main__":
    main()
