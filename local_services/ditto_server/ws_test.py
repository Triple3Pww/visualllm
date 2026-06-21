"""Headless smoke test for the Ditto server's websocket streaming path.

Sends the example audio as 16 kHz PCM and verifies RGB frames come back.
Run the server first (on the port below), then:
    conda run -n ditto python -m local_services.ditto_server.ws_test [ws_url]
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets

VENDOR = Path(__file__).resolve().parent / "vendor" / "ditto-talkinghead"
WAV = VENDOR / "example" / "audio.wav"
WS_URL = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8009/stream"


async def main():
    audio, sr = sf.read(str(WAV), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        n = int(round(len(audio) * 16000 / sr))
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)
    pcm16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)

    import time
    frames = 0
    first_shape = None
    t_first = None
    async with websockets.connect(WS_URL, max_size=None, ping_interval=None) as ws:
        await ws.send(json.dumps({"type": "config", "fps": 25}))
        await ws.send(json.dumps({"type": "speech_start"}))

        async def sender():
            # ~200 ms PCM chunks, like a TTS stream
            step = 3200
            for i in range(0, len(pcm16), step):
                await ws.send(pcm16[i:i + step].tobytes())
                await asyncio.sleep(0.05)
            await ws.send(json.dumps({"type": "speech_end"}))

        send_task = asyncio.create_task(sender())
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=8)
                if isinstance(msg, (bytes, bytearray)):
                    frames += 1
                    if first_shape is None:
                        first_shape = len(msg)
                        t_first = time.time()
                        print(f"first frame bytes={len(msg)} (expect 786432 for 512^2 RGB)")
                    if frames >= 200:
                        break
                else:
                    # Text sync markers (video_start / video_clock / video_end).
                    # Not part of the frame count; surfaced for sync debugging.
                    print(f"marker: {msg}")
        except asyncio.TimeoutError:
            pass
        send_task.cancel()

    elapsed = (time.time() - t_first) if t_first else 0
    fps = (frames - 1) / elapsed if elapsed > 0 else 0
    print(f"RECEIVED {frames} frames in {elapsed:.1f}s -> {fps:.1f} fps (realtime = 25)")
    print("PASS" if frames > 0 else "FAIL (no frames)")


if __name__ == "__main__":
    asyncio.run(main())
