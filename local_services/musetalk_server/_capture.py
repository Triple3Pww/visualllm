"""Throwaway: drive the MuseTalk server with a wav and mux frames+audio to mp4.

Lets you judge lip-sync offline (no WebRTC/mic). Run in the musetalk env after the
server is up:  python -m local_services.musetalk_server._capture output/cosy_en.wav
"""
import asyncio, json, subprocess, sys, wave
import numpy as np
import cv2
import websockets

URL = "ws://localhost:8002/stream"
SIZE = 256
FPS = 20
WAV = sys.argv[1] if len(sys.argv) > 1 else "output/cosy_en.wav"
STEM = WAV.rsplit("/", 1)[-1].rsplit(".", 1)[0]
OUT_SILENT = f"output/{STEM}_musetalk_silent.mp4"
OUT = f"output/{STEM}_musetalk.mp4"
FFMPEG = r"E:\miniconda3\envs\tts\Library\bin\ffmpeg.exe"


def load_16k(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate(); raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype=np.int16)
    if sr != 16000:
        n = int(len(a) * 16000 / sr)
        a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.int16)
    return a


async def main():
    audio16 = load_16k(WAV)
    frames = []
    async with websockets.connect(URL, max_size=None) as ws:
        async def reader():
            try:
                while True:
                    m = await ws.recv()
                    if isinstance(m, (bytes, bytearray)) and len(m) == SIZE * SIZE * 3:
                        frames.append(bytes(m))
            except Exception:
                pass
        rt = asyncio.create_task(reader())
        await ws.send(json.dumps({"type": "config", "fps": FPS}))
        await ws.send(json.dumps({"type": "speech_start"}))
        chunk = int(16000 * 0.02)
        for i in range(0, len(audio16), chunk):
            await ws.send(audio16[i:i + chunk].tobytes())
            await asyncio.sleep(0.02)
        await ws.send(json.dumps({"type": "speech_end"}))
        await asyncio.sleep(2.5)
        rt.cancel()

    audio_s = len(audio16) / 16000
    print(f"captured {len(frames)} frames = {len(frames)/FPS:.2f}s video for {audio_s:.2f}s audio")
    vw = cv2.VideoWriter(OUT_SILENT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (SIZE, SIZE))
    for fb in frames:
        arr = np.frombuffer(fb, dtype=np.uint8).reshape(SIZE, SIZE, 3)
        vw.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    vw.release()
    subprocess.run(
        [FFMPEG, "-y", "-i", OUT_SILENT, "-i", WAV, "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", OUT],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
