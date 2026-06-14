"""Websocket round-trip test against a running MuseTalk server.

Mimics local_services/musetalk_video.py: a background task reads frames
*concurrently* while audio is streamed in (the real client never blocks the
server's sender). Run after the server is up:
    python -m local_services.musetalk_server.ws_test
"""
from __future__ import annotations

import asyncio
import json
import time

import numpy as np
import websockets

URL = "ws://localhost:8002/stream"
SIZE = 512
FPS = 20


async def main():
    expected = SIZE * SIZE * 3
    stats = {"got": 0, "bad": 0}

    async with websockets.connect(URL, max_size=None) as ws:
        async def reader():
            try:
                while True:
                    msg = await ws.recv()
                    if isinstance(msg, (bytes, bytearray)):
                        stats["got"] += 1
                        if len(msg) != expected:
                            stats["bad"] += 1
            except Exception:  # noqa: BLE001
                pass

        rtask = asyncio.create_task(reader())

        await ws.send(json.dumps({"type": "config", "fps": FPS}))
        await ws.send(json.dumps({"type": "speech_start"}))

        sr = 16000
        rng = np.random.default_rng(1)
        audio = (3000 * rng.standard_normal(int(sr * 1.5))).astype(np.int16)
        chunk = int(sr * 0.02)
        for i in range(0, len(audio), chunk):
            await ws.send(audio[i:i + chunk].tobytes())
            await asyncio.sleep(0.02)  # stream at ~realtime
        await ws.send(json.dumps({"type": "speech_end"}))

        await asyncio.sleep(2.0)  # let remaining frames drain
        rtask.cancel()

    print(f"[ws_test] received {stats['got']} frames, {stats['bad']} wrong-sized "
          f"(expected {expected} bytes each)")
    assert stats["got"] > 5, "too few frames received"
    assert stats["bad"] == 0, "some frames had the wrong byte size"
    print("[ws_test] PASS")


if __name__ == "__main__":
    asyncio.run(main())
