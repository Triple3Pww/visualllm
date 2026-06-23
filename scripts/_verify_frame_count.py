"""Verify the avatar renders frames = audio_sec*fps at fps=12 (the frame-deficit fix).

Drives the LIVE musetalk server (:8002) directly over its websocket at fps=12 with a
known-duration wav, then reports frames received vs the audio length. The server also
logs `[stream] turn rendered N`. Pre-fix a ~13.5s reply rendered ~141 frames (lips
finished ~1.7s early); post-fix it should be ~162 (= floor(13.56*12)).

Run (system python, has websockets):  python -m scripts._verify_frame_count [wav] [fps]
"""
import asyncio, json, sys, wave
import numpy as np
import websockets

URL = "ws://localhost:8002/stream"
WAV = sys.argv[1] if len(sys.argv) > 1 else "output/reply_concise.wav"
FPS = int(sys.argv[2]) if len(sys.argv) > 2 else 12


def load_16k(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate(); raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype=np.int16)
    if sr != 16000:
        n = int(len(a) * 16000 / sr)
        a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.int16)
    return a


async def main():
    audio = load_16k(WAV)
    audio_s = len(audio) / 16000
    got = {"n": 0}
    async with websockets.connect(URL, max_size=None) as ws:
        async def reader():
            try:
                while True:
                    m = await ws.recv()
                    if isinstance(m, (bytes, bytearray)):
                        got["n"] += 1
            except Exception:
                pass
        rt = asyncio.create_task(reader())
        await ws.send(json.dumps({"type": "config", "fps": FPS}))
        await ws.send(json.dumps({"type": "speech_start"}))
        chunk = int(16000 * 0.04)
        for i in range(0, len(audio), chunk):
            await ws.send(audio[i:i + chunk].tobytes())
            await asyncio.sleep(0.005)   # feed fast; frame COUNT is fps/len-driven, not pacing
        await ws.send(json.dumps({"type": "speech_end"}))
        await asyncio.sleep(3.0)
        rt.cancel()

    want = int(np.floor(audio_s * FPS))
    print(f"audio={audio_s:.2f}s  fps={FPS}  want~{want} lip frames")
    print(f"frames RECEIVED by client (incl held/idle/tail): {got['n']}")
    print(f"-> check the server log line  [stream] turn rendered N  (N should be ~{want}, "
          f"not ~{int(want*7/8)})")


if __name__ == "__main__":
    asyncio.run(main())
