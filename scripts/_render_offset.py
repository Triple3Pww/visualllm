"""Offline render lip-offset: drive the MuseTalk server directly with a wav, read the
video_start/video_clock/video_end markers, collect the TURN's clean RGB frames, and measure
how far the mouth motion leads/lags the audio CONTENT (high-SNR, no WebRTC/VP8). This is the
RENDER's intrinsic alignment -> sets MUSETALK_SYNC_LEAD_S. Run in the musetalk env; STOP the
pipeline first (single-client server).  python -m scripts._render_offset output/q_ai.wav
"""
import asyncio, json, sys, wave
import numpy as np
import websockets

URL = "ws://localhost:8002/stream"
SIZE = 256
FPS = 20
WAV = sys.argv[1] if len(sys.argv) > 1 else "output/q_ai.wav"


def load(path):
    with wave.open(path, "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def to16k(a, sr):
    if sr == 16000:
        return a
    n = int(round(len(a) * 16000 / sr))
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a)


def xcorr(m, e, max_lag):
    n = min(len(m), len(e))
    m = (m[:n] - m[:n].mean()) / (m[:n].std() + 1e-9)
    e = (e[:n] - e[:n].mean()) / (e[:n].std() + 1e-9)
    best_lag, best = 0, -2.0
    for lag in range(-max_lag, max_lag + 1):
        mm, ee = (m[lag:], e[: n - lag]) if lag >= 0 else (m[: n + lag], e[-lag:])
        if len(mm) >= 8:
            c = float((mm * ee).mean())
            if c > best:
                best, best_lag = c, lag
    return best_lag, best


async def main():
    a24, sr = load(WAV)
    a16 = to16k(a24, sr)
    pcm = (np.clip(a16 / 32768.0, -1, 1) * 32767).astype(np.int16)
    frames, started, ended = [], False, False

    async with websockets.connect(URL, max_size=None, ping_interval=None) as ws:
        async def rx():
            nonlocal started, ended
            try:
                while True:
                    m = await ws.recv()
                    if isinstance(m, bytes):
                        if started and not ended and len(m) == SIZE * SIZE * 3:
                            frames.append(m)
                    else:
                        e = json.loads(m)
                        if e["type"] == "video_start":
                            started = True
                        elif e["type"] == "video_end":
                            ended = True
            except Exception:
                pass
        t = asyncio.create_task(rx())
        await ws.send(json.dumps({"type": "config", "fps": FPS}))
        await ws.send(json.dumps({"type": "speech_start"}))
        step = int(16000 * 0.02)
        for i in range(0, len(pcm), step):
            await ws.send(pcm[i:i + step].tobytes())
            await asyncio.sleep(0.02)
        await ws.send(json.dumps({"type": "speech_end"}))
        await asyncio.sleep(3.0)
        t.cancel()

    n = len(frames)
    print(f"captured {n} turn frames for {len(a16)/16000:.1f}s audio")
    if n < 12:
        print("too few frames"); return
    arr = np.stack([np.frombuffer(f, np.uint8).reshape(SIZE, SIZE, 3) for f in frames]).astype(np.float64)
    # mouth box = lower-face region of highest temporal motion
    lower = arr[:, SIZE // 2:, :, :]
    mot_map = np.abs(np.diff(lower.mean(axis=3), axis=0)).mean(axis=0)
    rr, cc = np.unravel_index(int(np.argmax(mot_map)), mot_map.shape)
    r0 = SIZE // 2 + max(0, rr - SIZE // 8); r1 = min(SIZE, r0 + SIZE // 4)
    c0 = max(0, cc - SIZE // 8); c1 = min(SIZE, c0 + SIZE // 4)
    region = arr[:, r0:r1, c0:c1, :].mean(axis=3)
    mot = np.concatenate([[0.0], np.abs(np.diff(region, axis=0)).mean(axis=(1, 2))])
    # audio RMS per output frame
    env = np.zeros(n)
    for i in range(n):
        s = int(i / FPS * 16000); e = int((i + 1) / FPS * 16000)
        if e <= len(a16) and e > s:
            env[i] = np.sqrt(np.mean((a16[s:e] / 32768.0) ** 2) + 1e-12)
    lag, corr = xcorr(mot, env, max_lag=FPS)
    off = lag / FPS
    sign = "mouth LAGS audio" if off > 0 else "mouth LEADS audio"
    print(f"RENDER lip offset: {off*1000:+.0f}ms ({sign}), corr={corr:.2f} "
          f"[{'reliable' if corr > 0.3 else 'low-conf'}]")
    print(f"-> suggested MUSETALK_SYNC_LEAD_S = {off:+.2f}  (cancels the render's intrinsic offset)")


if __name__ == "__main__":
    asyncio.run(main())
