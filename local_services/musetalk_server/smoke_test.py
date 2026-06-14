"""Standalone smoke test for the MuseTalk engine (no websocket/pipeline).

Run in the musetalk env:
    conda run -n musetalk python -m local_services.musetalk_server.smoke_test

Verifies: models load, the avatar prepares (face detected + cached), and a
segment of audio renders to frames of exactly IMAGE_SIZE*IMAGE_SIZE*3 bytes.
"""
from __future__ import annotations

import time

import numpy as np

from local_services.musetalk_server import app


def main():
    t0 = time.time()
    app.engine.load()
    print(f"[smoke] load() OK in {time.time() - t0:.1f}s; "
          f"base frames={len(app.engine.frame_cycle)} size={app.engine.size}")

    expected = app.IMAGE_SIZE * app.IMAGE_SIZE * 3
    rng = np.random.default_rng(0)
    spf = app.engine.samples_per_frame(app.engine.fps)
    seg = (0.05 * rng.standard_normal(spf * app.SEG_FRAMES)).astype(np.float32)  # one stream segment

    # First call pays one-time CUDA kernel autotune/JIT (Blackwell). Warm up.
    print("[smoke] warming up (first call pays kernel autotune) …")
    t0 = time.time()
    frames = app.engine.render_segment(seg)
    print(f"[smoke] warmup: {len(frames)} frames in {(time.time()-t0)*1000:.0f}ms")

    assert frames, "no frames produced"
    bad = [len(f) for f in frames if len(f) != expected]
    assert not bad, f"wrong frame byte size (expected {expected}); got {bad[:3]}"
    assert len(app.engine.neutral_frame()) == expected, "neutral frame wrong size"

    # Warm throughput over several stream-sized segments.
    N = 8
    t1 = time.time()
    total = 0
    for _ in range(N):
        total += len(app.engine.render_segment(seg))
    dt = time.time() - t1
    fps = total / max(dt, 1e-6)
    ms_seg = dt / N * 1000
    realtime = app.SEG_FRAMES / app.engine.fps * 1000  # wall-time budget per segment
    print(f"[smoke] WARM: {total} frames over {N} segments, {fps:.1f} fps "
          f"({ms_seg:.0f}ms/seg vs {realtime:.0f}ms realtime budget)")
    print(f"[smoke] realtime margin: {realtime/ms_seg:.2f}x  "
          f"({'OK' if ms_seg < realtime else 'TOO SLOW'})")
    print(f"[smoke] all frames == {expected} bytes. PASS")


if __name__ == "__main__":
    main()
