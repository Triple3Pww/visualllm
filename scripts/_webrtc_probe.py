"""Headless WebRTC probe — measure the live avatar (freeze / sync / smoothness) with no mic.

Connects to the running pipeline at http://localhost:7860/api/offer like a browser, sends a
question wav as the mic (with leading silence so it lands AFTER the greeting, past echo-guard),
and RECEIVES the bot's audio+video. Records a correctly-A/V-timed mp4 (aiortc MediaRecorder)
and prints metrics that map to the avatar gripes:

  freeze        -> max wall-clock gap between received video frames
  choppy        -> received inter-frame interval mean/jitter
  voice stutter -> max gap in received audio arrival
  startup       -> connect -> first received video frame
  lip offset    -> mouth-motion x audio-RMS xcorr on the RECORDED mp4 (container A/V clock);
                   +ve = lips lag voice, -ve = lead; corr>~0.3 = trustworthy

Run (pipeline + both servers up):  python -m scripts._webrtc_probe --mic output/q_ai.wav --lead 10
"""
from __future__ import annotations

import argparse
import asyncio
import time
import wave
from pathlib import Path

import aiohttp
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer, MediaRecorder, MediaRelay

ROOT = Path(__file__).resolve().parent.parent
OFFER_URL = "http://127.0.0.1:7860/api/offer"
MP4 = str(ROOT / "output" / "probe_live.mp4")

VWALL: list[float] = []   # wall-clock arrival of each video frame
AWALL: list[float] = []   # wall-clock arrival of each audio packet


def build_mic_wav(src: str, lead_s: float, tail_s: float) -> str:
    with wave.open(src, "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if ch > 1:
        a = a.reshape(-1, ch)[:, 0]
    out = np.concatenate([np.zeros(int(lead_s * sr), np.int16), a,
                          np.zeros(int(tail_s * sr), np.int16)])
    dst = str(ROOT / "output" / "_mic_drive.wav")
    with wave.open(dst, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(out.tobytes())
    return dst


async def count_video(track):
    while True:
        try:
            await track.recv()
        except Exception:
            return
        VWALL.append(time.time())


async def count_audio(track):
    while True:
        try:
            await track.recv()
        except Exception:
            return
        AWALL.append(time.time())


async def wait_ice(pc):
    if pc.iceGatheringState == "complete":
        return
    done = asyncio.Event()
    @pc.on("icegatheringstatechange")
    def _():
        if pc.iceGatheringState == "complete":
            done.set()
    await done.wait()


# --------------------------------------------------------------- lip-offset from the mp4
def _auto_mouth_box(arr, h, w):
    lower = arr[:, h // 2:, :, :]
    mot = np.abs(np.diff(lower.mean(axis=3), axis=0)).mean(axis=0)
    rr, cc = np.unravel_index(int(np.argmax(mot)), mot.shape)
    r0 = h // 2 + max(0, rr - h // 8); r1 = min(h, r0 + h // 4)
    c0 = max(0, cc - w // 8); c1 = min(w, c0 + w // 4)
    return r0, r1, c0, c1


def _xcorr_lag(m, a, max_lag):
    n = min(len(m), len(a))
    if n < 8:
        return 0, 0.0
    m = (m[:n] - m[:n].mean()) / (m[:n].std() + 1e-9)
    a = (a[:n] - a[:n].mean()) / (a[:n].std() + 1e-9)
    best_lag, best = 0, -2.0
    for lag in range(-max_lag, max_lag + 1):
        mm, aa = (m[lag:], a[: n - lag]) if lag >= 0 else (m[: n + lag], a[-lag:])
        if len(mm) >= 8:
            c = float((mm * aa).mean())
            if c > best:
                best, best_lag = c, lag
    return best_lag, best


def lip_offset_from_mp4(path: str, fps: float):
    """A/V on the SAME container clock: video frame i at vpts[i], audio sample-accurate.
    Mouth-motion (per frame) vs audio-RMS over each frame window, windowed to the talking
    region, cross-correlated."""
    import av
    cont = av.open(path)
    if not cont.streams.video:
        cont.close(); return None, 0.0, "no video in mp4"
    vframes, vpts = [], []
    for f in cont.decode(video=0):
        vframes.append(f.to_ndarray(format="rgb24"))
        vpts.append(float((f.pts or 0) * f.time_base))
    cont.close()
    cont = av.open(path)
    asr, asamps, a0 = None, [], None
    for f in cont.decode(audio=0):
        if asr is None:
            asr = f.sample_rate; a0 = float((f.pts or 0) * f.time_base)
        s = f.to_ndarray()
        asamps.append((s.mean(axis=0) if s.ndim == 2 else s).astype(np.float64))
    cont.close()
    if len(vframes) < 10 or not asamps:
        return None, 0.0, "too few frames"
    frames = np.stack(vframes)
    vt = np.array(vpts) - vpts[0]
    H, W = frames.shape[1], frames.shape[2]
    r0, r1, c0, c1 = _auto_mouth_box(frames, H, W)
    region = frames[:, r0:r1, c0:c1, :].mean(axis=3)
    mot = np.concatenate([[0.0], np.abs(np.diff(region, axis=0)).mean(axis=(1, 2))])
    aud = np.concatenate(asamps)
    if np.abs(aud).max() > 2:
        aud = aud / 32768.0
    astart = (a0 or 0.0) - vpts[0]
    env = np.zeros(len(vt))
    for i, t in enumerate(vt):
        s = int((t - astart) * asr); e = int((t + 1.0 / fps - astart) * asr)
        s, e = max(0, s), min(len(aud), e)
        if e > s:
            env[i] = float(np.sqrt(np.mean(aud[s:e] ** 2) + 1e-12))
    if env.max() > 0:
        idx = np.where(env > 0.18 * env.max())[0]
        if len(idx) > 8:
            lo, hi = max(0, idx[0] - int(fps)), min(len(env), idx[-1] + int(fps))
            mot, env = mot[lo:hi], env[lo:hi]
    lag, corr = _xcorr_lag(mot, env, max_lag=int(fps))
    return lag / fps, corr, None


# ----------------------------------------------------------------------------------- report
def report(fps: float, connect_t: float):
    print("\n==================== WEBRTC PROBE REPORT ====================")
    print(f"video frames: {len(VWALL)}   audio packets: {len(AWALL)}")
    if len(VWALL) < 5:
        print("NO/while VIDEO RECEIVED — avatar produced almost no frames."); return
    walls = np.array(VWALL)
    gaps = np.diff(walls)
    print(f"startup (connect -> first video frame): {walls[0]-connect_t:.2f}s")
    print(f"received fps (wall): {len(VWALL)/(walls[-1]-walls[0]+1e-9):.1f}")
    print(f"frame interval ms: mean={gaps.mean()*1000:.1f} p95={np.percentile(gaps,95)*1000:.1f} "
          f"max={gaps.max()*1000:.1f}  (target ~{1000/fps:.0f}; choppy if p95>>{1000/fps:.0f})")
    print(f"FREEZE: max gap between frames = {gaps.max()*1000:.0f}ms "
          f"({'FAIL >500ms' if gaps.max()>0.5 else 'ok'})")
    if len(AWALL) > 2:
        ag = np.diff(np.array(AWALL))
        print(f"audio arrival gap ms: p95={np.percentile(ag,95)*1000:.1f} max={ag.max()*1000:.1f} "
              f"({'FAIL >50ms' if ag.max()>0.05 else 'ok'})")
    off, corr, err = lip_offset_from_mp4(MP4, fps)
    if err:
        print(f"LIP OFFSET: unavailable ({err})")
    elif off is not None:
        sign = "lips LAG voice" if off > 0 else "lips LEAD voice"
        verdict = "ok" if abs(off) < 0.08 else "FAIL >80ms"
        rel = "reliable" if corr > 0.3 else "LOW-CONF (corr<0.3)"
        print(f"LIP OFFSET: {off*1000:+.0f}ms ({sign}), corr={corr:.2f} [{rel}] ({verdict})")
    print(f"recorded mp4 (watch/listen to judge): {MP4}")
    print("============================================================\n")


async def main(args):
    mic = build_mic_wav(args.mic, args.lead, args.tail) if args.mic else None
    pc = RTCPeerConnection()
    if mic:
        pc.addTrack(MediaPlayer(mic).audio)
    else:
        pc.addTransceiver("audio", direction="recvonly")
    pc.addTransceiver("video", direction="recvonly")

    tracks: dict = {}

    @pc.on("track")
    def on_track(track):
        tracks[track.kind] = track

    await pc.setLocalDescription(await pc.createOffer())
    await wait_ice(pc)
    connect_t = time.time()
    async with aiohttp.ClientSession() as s:
        async with s.post(OFFER_URL, json={"sdp": pc.localDescription.sdp, "type": "offer"}) as r:
            ans = await r.json()
    await pc.setRemoteDescription(RTCSessionDescription(sdp=ans["sdp"], type=ans["type"]))

    # wait for both inbound tracks, THEN wire recorder + metrics via a relay (so both see frames)
    for _ in range(50):
        if "video" in tracks and "audio" in tracks:
            break
        await asyncio.sleep(0.1)
    relay = MediaRelay()
    recorder = MediaRecorder(MP4)
    if "video" in tracks:
        recorder.addTrack(relay.subscribe(tracks["video"]))
        asyncio.ensure_future(count_video(relay.subscribe(tracks["video"])))
    if "audio" in tracks:
        recorder.addTrack(relay.subscribe(tracks["audio"]))
        asyncio.ensure_future(count_audio(relay.subscribe(tracks["audio"])))
    await recorder.start()
    print(f"connected (pc_id={ans.get('pc_id')}, tracks={list(tracks)}); capturing {args.duration}s...")
    await asyncio.sleep(args.duration)
    await recorder.stop()
    await pc.close()
    report(args.fps, connect_t)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mic", default=None)
    ap.add_argument("--lead", type=float, default=10.0)
    ap.add_argument("--tail", type=float, default=28.0)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--fps", type=float, default=20.0)
    asyncio.run(main(ap.parse_args()))
