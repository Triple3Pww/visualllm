"""Capture the LIVE real-time avatar stream to an mp4 (a faithful 'what the browser
would show' recording, minus WebRTC/RDP).

Unlike ditto_offline.py (full-quality, whole-clip offline render), this drives the
*running* websocket server exactly like the pipeline does: config fps, speech_start,
16 kHz PCM streamed in real time, frames received at the server's pump rate (with the
real frame-drop + any held/stalled frames). It then encodes the received frames at the
output fps and muxes the audio -- so the mp4 reproduces the real-time smoothness. If this
file plays smooth but the RDP/browser view lags, the lag is the viewing path, not the avatar.

Run (server must be up, and NOT have another client -- stop the pipeline first):
    E:\\miniconda3\\envs\\ditto\\python.exe -m local_services.ditto_server.capture_mp4 \
        --wav output/what_is_ai.wav --out output/realtime_capture.mp4 --fps 12
ASCII-only.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets

ROOT = Path(__file__).resolve().parents[2]
FFMPEG = (
    "C:/Users/MARU/AppData/Local/Microsoft/WinGet/Packages/"
    "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1.1-full_build/bin/ffmpeg.exe"
)
import os
# Must match the avatar server's DITTO_SIZE (the frame the ws actually streams); a
# fixed 512 here mismatches a 320/384 server and feeds ffmpeg misaligned rawvideo.
SIZE = int(os.getenv("DITTO_SIZE", "512") or "512")


async def capture(url: str, wav: Path, fps: int) -> list[bytes]:
    audio, sr = sf.read(str(wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        n = int(round(len(audio) * 16000 / sr))
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)
    pcm16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()

    frames: list[bytes] = []
    started = False  # collect frames only from video_start (drop leading idle frames)
    done = asyncio.Event()
    import time as _t
    t_speech = None
    t_vstart = None

    async with websockets.connect(url, max_size=None, ping_interval=None) as ws:
        await ws.send(json.dumps({"type": "config", "fps": fps}))
        t_speech = _t.time()
        await ws.send(json.dumps({"type": "speech_start"}))

        async def sender():
            chunk = 16000 * 2 // 12  # ~80ms of PCM per send, real-time paced
            for i in range(0, len(pcm16), chunk):
                await ws.send(pcm16[i:i + chunk])
                await asyncio.sleep(0.06)
            await ws.send(json.dumps({"type": "speech_end"}))

        send_task = asyncio.create_task(sender())
        idle_after_send = 0.0
        try:
            while not done.is_set():
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=10)
                except asyncio.TimeoutError:
                    break
                if isinstance(msg, bytes):
                    if started:
                        frames.append(msg)
                elif isinstance(msg, str):
                    evt = json.loads(msg)
                    t = evt.get("type")
                    if t == "video_start":
                        started = True
                        if t_vstart is None:
                            t_vstart = _t.time()
                            print(f"START LATENCY: video_start {t_vstart - t_speech:.2f}s after speech_start")
                    elif t == "video_end":
                        # one full turn captured
                        if send_task.done():
                            break
        finally:
            send_task.cancel()
    return frames


def encode(frames: list[bytes], wav: Path, out: Path, fps: int, lead: float = 0.0) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    # lead > 0 = the mouth renders AHEAD of the voice by `lead` s (Ditto's look-ahead).
    # Delay the video by `lead` so the mouth shape lands ON its sound -> A/V aligned.
    # This mirrors the live client's DITTO_SYNC_LEAD_S compensation so the mp4 shows
    # what the corrected live stream looks like.
    cmd = [
        FFMPEG, "-loglevel", "error", "-y",
        "-itsoffset", f"{lead}",         # delay the video input by `lead` seconds
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIZE}x{SIZE}", "-r", str(fps),
        "-i", "-",                       # raw RGB frames on stdin
        "-i", str(wav),                  # the audio
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
        "-c:a", "aac", "-shortest",
        str(out),
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f)
    p.stdin.close()
    p.wait()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8002/stream")
    ap.add_argument("--wav", default=str(ROOT / "output" / "what_is_ai.wav"))
    ap.add_argument("--out", default=str(ROOT / "output" / "realtime_capture.mp4"))
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--lead", type=float, default=0.0,
                    help="seconds the mouth leads the voice; delays video to align (sync fix)")
    args = ap.parse_args()

    frames = asyncio.run(capture(args.url, Path(args.wav), args.fps))
    if not frames:
        print("NO FRAMES captured (is the server up and free of other clients?)")
        return 1
    dur = len(frames) / args.fps
    print(f"captured {len(frames)} frames -> {dur:.1f}s of video at {args.fps}fps (lead={args.lead}s)")
    encode(frames, Path(args.wav), Path(args.out), args.fps, args.lead)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
