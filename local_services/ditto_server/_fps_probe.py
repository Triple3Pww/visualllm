"""Throughput probe: feed a long continuous speech stream and report sustained
render fps. Temporary measurement tool (not part of the server)."""
import asyncio, json, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
import websockets

VENDOR = Path(__file__).resolve().parent / "vendor" / "ditto-talkinghead"
WAV = VENDOR / "example" / "audio.wav"
URL = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8002/stream"
REPEAT = int(sys.argv[2]) if len(sys.argv) > 2 else 6  # ~Nx the clip = long turn
FPS = int(sys.argv[3]) if len(sys.argv) > 3 else 12  # MUST match the real operating fps


async def main():
    audio, sr = sf.read(str(WAV), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        n = int(round(len(audio) * 16000 / sr))
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)
    audio = np.tile(audio, REPEAT)
    pcm16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()

    frames = 0
    t_first = None
    async with websockets.connect(URL, max_size=None, ping_interval=None) as ws:
        await ws.send(json.dumps({"type": "config", "fps": FPS}))
        await ws.send(json.dumps({"type": "speech_start"}))
        # stream PCM in ~80ms chunks at realtime-ish pace so the session stays "speaking"
        chunk = 16000 * 2 // 12
        async def sender():
            for i in range(0, len(pcm16), chunk):
                await ws.send(pcm16[i:i + chunk])
                await asyncio.sleep(0.06)
            await ws.send(json.dumps({"type": "speech_end"}))
        send_task = asyncio.create_task(sender())
        t0 = time.time()
        while time.time() - t0 < 40:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=8)
            except asyncio.TimeoutError:
                break
            if isinstance(msg, bytes):
                frames += 1
                if t_first is None:
                    t_first = time.time()
        send_task.cancel()
    elapsed = (time.time() - t_first) if t_first else 0
    print(f"PROBE received {frames} frames in {elapsed:.1f}s -> {frames/elapsed if elapsed else 0:.1f} fps received")


asyncio.run(main())
