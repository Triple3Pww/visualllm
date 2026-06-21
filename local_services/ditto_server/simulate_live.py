"""Simulate the LIVE avatar experience as an mp4 (idle warmup -> synced speech).

capture_mp4.py drops the warmup and shows only the spoken part. This one reproduces the
FULL live timeline the FIXED transport delivers, so you can judge sync the way it actually
plays in the browser -- minus WebRTC/RDP:

  * It keeps the IDLE frames the server streams before video_start (the ~2s diffusion
    warmup), so the mp4 opens on the living idle face (blink/breathe), exactly like live.
  * From video_start it shows the real lip-synced turn frames. With the transport fix
    (video_out_is_live=False + sync_with_audio) each turn frame displays at audio-time
    k/fps -- so here we encode every frame at a constant fps and DELAY the audio by the
    warmup length (frames-before-video_start / fps). Turn frame k then lands on audio
    sample k/fps = synced, while the warmup plays the idle face over leading silence.

This is the faithful "what live looks like now" artifact. Server must be up and free of
other clients (stop the pipeline first). Run in the ditto env:
    E:\\miniconda3\\envs\\ditto\\python.exe -m local_services.ditto_server.simulate_live \
        --wav output/tune_clips/vlong.wav --out output/live_sim.mp4 --fps 12
ASCII-only.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets

ROOT = Path(__file__).resolve().parents[2]
FFMPEG = (
    "C:/Users/MARU/AppData/Local/Microsoft/WinGet/Packages/"
    "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1.1-full_build/bin/ffmpeg.exe"
)
SIZE = 512


async def capture(url: str, wav: Path, fps: int) -> tuple[list[bytes], int]:
    """Drive one real turn; return ALL frames (idle warmup + turn) and the index of the
    first turn frame (== frames received before video_start = the warmup length)."""
    audio, sr = sf.read(str(wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        n = int(round(len(audio) * 16000 / sr))
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)
    pcm16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()

    frames: list[bytes] = []
    vstart_index: int | None = None
    t_speech = None

    async with websockets.connect(url, max_size=None, ping_interval=None) as ws:
        await ws.send(json.dumps({"type": "config", "fps": fps}))
        # Let the idle warmup stream for ~1.5s so the mp4 opens on the living idle face
        # (the server pumps idle frames immediately once the session is ready).
        warm_until = time.time() + 1.5
        while time.time() < warm_until:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=2)
            except asyncio.TimeoutError:
                break
            if isinstance(msg, bytes):
                frames.append(msg)

        t_speech = time.time()
        await ws.send(json.dumps({"type": "speech_start"}))

        async def sender():
            chunk = 16000 * 2 // 12  # ~80ms PCM per send, real-time paced
            for i in range(0, len(pcm16), chunk):
                await ws.send(pcm16[i:i + chunk])
                await asyncio.sleep(0.06)
            await ws.send(json.dumps({"type": "speech_end"}))

        send_task = asyncio.create_task(sender())
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=12)
                except asyncio.TimeoutError:
                    break
                if isinstance(msg, bytes):
                    frames.append(msg)
                elif isinstance(msg, str):
                    evt = json.loads(msg)
                    if evt.get("type") == "video_start" and vstart_index is None:
                        vstart_index = len(frames)  # frames so far = the warmup shown
                        print(f"warmup = {vstart_index} frames "
                              f"({vstart_index / fps:.2f}s); video_start "
                              f"{time.time() - t_speech:.2f}s after speech_start")
                    elif evt.get("type") == "video_end" and send_task.done():
                        break
        finally:
            send_task.cancel()
    if vstart_index is None:
        vstart_index = 0
    return frames, vstart_index


def encode(frames: list[bytes], wav: Path, out: Path, fps: int, audio_delay_s: float) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    # Delay the audio by PREPENDING real silence (ffmpeg -itsoffset silently no-ops with
    # some muxers -- it left the voice starting at frame 0). Building the delayed wav in
    # numpy is unambiguous: the voice then begins exactly at the first turn frame, so turn
    # frame k lands on audio k/fps (synced) and the warmup plays over leading silence.
    a, sr = sf.read(str(wav), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    pad = np.zeros(int(round(audio_delay_s * sr)), dtype=np.float32)
    delayed = Path(str(out) + ".audio.wav")
    sf.write(str(delayed), np.concatenate([pad, a]), sr)
    # Video: all frames at constant fps (idle warmup then the turn).
    cmd = [
        FFMPEG, "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIZE}x{SIZE}", "-r", str(fps),
        "-i", "-",                          # raw RGB frames on stdin
        "-i", str(delayed),                 # audio with the warmup silence baked in
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
        "-c:a", "aac",
        str(out),
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f)
    p.stdin.close()
    p.wait()
    delayed.unlink(missing_ok=True)


def _verify_raw(turn_frames: list[bytes], wav: Path, fps: int) -> None:
    """Measure mouth-motion vs audio-RMS offset on the RAW turn frames (avatar_tune's
    method, no h264 noise). Prints the intrinsic A/V offset of what's in the mp4."""
    try:
        from scripts.avatar_tune import _openness_env, _audio_env
    except Exception as e:  # noqa: BLE001
        print(f"(skip raw verify: {e})")
        return
    if len(turn_frames) < 10:
        print("(skip raw verify: too few turn frames)")
        return
    a, sr = sf.read(str(wav), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    spf = max(1, round(16000 / fps)) * 1  # avatar_tune uses round(25/fps)*640; keep its spf
    from scripts.avatar_tune import _load_16k
    pcm = _load_16k(wav)
    spf = max(1, round(25 / fps)) * 640
    arms = _audio_env(pcm, spf)
    # avatar_tune.align method: openness env (|region - resting median|) + 3-tap smoothing.
    op = _openness_env(turn_frames, SIZE)

    def sm(x, k=3):
        return np.convolve(x, np.ones(k) / k, mode="same")

    def z(x):
        return (x - x.mean()) / (x.std() + 1e-9)

    n = min(len(op), len(arms))
    zo, za = z(sm(op[:n])), z(sm(arms[:n]))
    best_lag, best = 0, -2.0
    for lag in range(-15, 16):
        if lag >= 0:
            mm, aa = zo[lag:], za[:n - lag]
        else:
            mm, aa = zo[:n + lag], za[-lag:]
        if len(mm) < 8:
            continue
        c = float((mm * aa).mean())
        if c > best:
            best, best_lag = c, lag
    verdict = "SYNCED" if abs(best_lag / fps) <= 0.12 else f"offset {best_lag/fps:+.2f}s"
    print(f"raw-frame sync check (openness/align method): {best_lag/fps:+.2f}s "
          f"(corr {best:+.2f}) -> {verdict}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8002/stream")
    ap.add_argument("--wav", default=str(ROOT / "output" / "tune_clips" / "vlong.wav"))
    ap.add_argument("--out", default=str(ROOT / "output" / "live_sim.mp4"))
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    frames, vstart_index = asyncio.run(capture(args.url, Path(args.wav), args.fps))
    if not frames:
        print("NO FRAMES captured (is the server up and free of other clients?)")
        return 1

    # Verify sync on the RAW turn frames (pre-h264, so no encode noise) the same way
    # avatar_tune does: the turn frames [vstart:] aligned with the audio from 0. ~0 here
    # means the encoded mp4 (same frames at fps + audio delayed by the warmup) is synced.
    _verify_raw(frames[vstart_index:], Path(args.wav), args.fps)

    audio_delay = vstart_index / args.fps
    dur = len(frames) / args.fps
    print(f"captured {len(frames)} frames -> {dur:.1f}s video; audio delayed {audio_delay:.2f}s "
          f"(warmup) so the voice starts on the first synced lip frame")
    encode(frames, Path(args.wav), Path(args.out), args.fps, audio_delay)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
