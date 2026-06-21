"""Autonomous avatar lip-sync tuning harness.

Drives the Ditto ws server EXACTLY like the real client (local_services/ditto_video.py)
-- config{fps} -> speech_start -> real-time-paced 16k PCM -> speech_end, reading the
video_start/video_clock/video_end markers -- but ALSO keeps the returned RGB frames and
measures the thing the `hold` trace can't: **does the mouth actually track the voice.**

Primary metric -- intrinsic lip alignment (`lip_lag_s`):
  For the rendered frames, build a per-frame MOUTH-MOTION envelope (mean abs diff of the
  lower-face region vs the resting face). Build an AUDIO-RMS envelope of the clip resampled
  to the output frame rate. Cross-correlate them: the lag that maximises correlation is how
  far the mouth motion leads/lags the audio CONTENT. That offset is a property of the render
  (windowing/model), independent of playback pacing -- and the correct DITTO_SYNC_LEAD_S is
  simply its negation (release the voice to cancel the render's lead/lag). So we MEASURE the
  offset instead of blindly sweeping the knob.

Secondary (playback timing, from markers): render_wait, tail, max|hold|, real_fps.

Runs in the 'ditto' conda env (needs numpy/soundfile/websockets). It manages the server
itself (kill + relaunch with env overrides), so run ONE instance at a time.

    conda run -n ditto python -m scripts.avatar_tune clips
    conda run -n ditto python -m scripts.avatar_tune measure --set "OVERLAP=10,FPS=12.5,PRERENDER=1,LEAD=0.2"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets

ROOT = Path(__file__).resolve().parent.parent
DITTO_PY = Path(r"E:\miniconda3\envs\ditto\python.exe")
SRC_WAV = ROOT / "output" / "what_is_ai.wav"
CLIP_DIR = ROOT / "output" / "tune_clips"
JSONL = ROOT / "logs" / "tune.jsonl"
WS = "ws://localhost:8002/stream"
HEALTH = "http://127.0.0.1:8002/health"
STATUS = "http://127.0.0.1:8002/status"

# The clip battery: (name, start_s, dur_s) carved from the 53s narration. Lengths chosen
# to probe short (render-wait), medium + long (robust xcorr), back-to-back (multi-turn).
CLIP_SPEC = [
    ("short", 7.0, 1.6),
    ("medium", 10.0, 3.2),
    ("long", 20.0, 6.0),
    ("vlong", 5.0, 18.0),   # continuous speech, for robust intrinsic A/V-offset xcorr
]


# ----------------------------------------------------------------------------- clips
def _make_calib(audio: np.ndarray, sr: int) -> tuple[np.ndarray, list[float]]:
    """Calibration clip: a loud speech syllable separated by silence, repeated. The
    silences let the mouth return to rest so each syllable is a SHARP mouth-open onset ->
    a clean, unambiguous A/V offset measurement (vs muddy continuous speech). Returns the
    clip and the list of syllable ONSET times (seconds)."""
    win = int(0.40 * sr)
    # loudest 0.40s window = a clear consonant+vowel
    rms = np.sqrt(np.convolve(audio.astype(np.float64) ** 2, np.ones(win) / win, "valid"))
    peak = int(np.argmax(rms))
    syll = audio[max(0, peak - int(0.05 * sr)): peak - int(0.05 * sr) + win]
    # NON-periodic, LONG gaps: long enough (>=1.3s) that the mouth fully closes between
    # bursts (no onset contamination), and varied so xcorr has no periodic-aliasing peak.
    gaps = [1.5, 1.4, 1.6, 1.35, 1.55, 1.45]
    clip = []
    onsets = []
    t = 0.0
    for g in gaps:
        sil = np.zeros(int(g * sr), np.float32)
        clip.append(sil); t += g
        onsets.append(t)
        clip.append(syll); t += len(syll) / sr
    clip.append(np.zeros(int(1.3 * sr), np.float32))
    return np.concatenate(clip).astype(np.float32), onsets


def make_clips() -> dict[str, Path]:
    audio, sr = sf.read(str(SRC_WAV), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, t0, dur in CLIP_SPEC:
        seg = audio[int(t0 * sr): int((t0 + dur) * sr)]
        p = CLIP_DIR / f"{name}.wav"
        sf.write(str(p), seg, sr)
        out[name] = p
    calib, onsets = _make_calib(audio, sr)
    sf.write(str(CLIP_DIR / "calib.wav"), calib, sr)
    (CLIP_DIR / "calib_onsets.json").write_text(json.dumps(onsets))
    out["calib"] = CLIP_DIR / "calib.wav"
    # Multi-sentence reply: 3 continuous-speech sentences with silence gaps -> the harness
    # drives speech_start/end PER sentence (like the real pipeline), exercising per-sentence
    # start-latency, the inter-sentence behavior, and the long-reply path that broke
    # compute-first. Saved as one wav PLUS the sentence (start,dur) spans for the driver.
    spans = [(25.0, 2.0), (30.0, 3.0), (38.0, 2.0)]
    sents, gap = [], np.zeros(int(0.6 * sr), np.float32)
    sent_spans = []  # (start_s, dur_s) within the saved multi.wav
    t = 0.0
    for s0, dur in spans:
        seg = audio[int(s0 * sr): int((s0 + dur) * sr)]
        sent_spans.append((round(t, 3), round(len(seg) / sr, 3)))
        sents.append(seg); sents.append(gap)
        t += len(seg) / sr + len(gap) / sr
    sf.write(str(CLIP_DIR / "multi.wav"), np.concatenate(sents).astype(np.float32), sr)
    (CLIP_DIR / "multi_spans.json").write_text(json.dumps(sent_spans))
    out["multi"] = CLIP_DIR / "multi.wav"
    print(f"clips -> {CLIP_DIR} ({', '.join(out)}); calib onsets={[round(o,2) for o in onsets]}; "
          f"multi spans={sent_spans}")
    return out


def _load_16k(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        n = int(round(len(audio) * 16000 / sr))
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)
    return audio.astype(np.float32)


# ----------------------------------------------------------------------- server mgmt
def _kill_server() -> None:
    # Stop any python running the ditto server (single-client slot must be free).
    subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and "
         "$_.CommandLine -match 'ditto_server\\.app' } | ForEach-Object { Stop-Process "
         "-Id $_.ProcessId -Force }"],
        capture_output=True, text=True,
    )
    time.sleep(3)


def _start_server(overrides: dict[str, str]) -> subprocess.Popen:
    import os

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Map our short knob names to the server's env vars.
    keymap = {"OVERLAP": "DITTO_OVERLAP", "FPS": "DITTO_FPS", "PRERENDER": "DITTO_PRERENDER",
              "STEPS": "DITTO_STEPS", "SIZE": "DITTO_SIZE", "LEAD": "DITTO_LEAD_FRAMES",
              "IDLE": "DITTO_IDLE_GRACE", "FLUSH": "DITTO_FLUSH_CHUNKS"}
    for k, v in overrides.items():
        if k in keymap:
            env[keymap[k]] = str(v)
    return subprocess.Popen(
        [str(DITTO_PY), "-u", "-m", "local_services.ditto_server.app"],
        cwd=str(ROOT), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _wait_health(timeout: float = 120) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(HEALTH, timeout=2) as r:
                d = json.loads(r.read())
                if d.get("ok"):
                    return d
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    raise RuntimeError("server did not become healthy")


# --------------------------------------------------------------------- envelopes
def _audio_env(pcm: np.ndarray, samples_per_frame: int) -> np.ndarray:
    """Per-output-frame audio RMS envelope."""
    n = len(pcm) // samples_per_frame
    if n == 0:
        return np.zeros(0, np.float32)
    w = pcm[: n * samples_per_frame].reshape(n, samples_per_frame)
    return np.sqrt((w.astype(np.float64) ** 2).mean(axis=1)).astype(np.float32)


def _frames_arr(frames: list[bytes], size: int) -> np.ndarray:
    return np.stack([np.frombuffer(f, np.uint8).reshape(size, size, 3) for f in frames]).astype(np.float32)


def _auto_mouth_box(arr: np.ndarray, size: int) -> tuple[int, int, int, int]:
    """Locate the mouth as the lower-face region of highest TEMPORAL motion. arr is
    (N,H,W,3). Returns (r0,r1,c0,c1)."""
    mot = np.abs(np.diff(arr, axis=0)).mean(axis=(0, 3))  # (H,W) per-pixel motion
    # restrict to the lower-middle face so eyes/blinks/head-top don't win
    mask = np.zeros_like(mot)
    r0, r1 = int(size * 0.45), int(size * 0.95)
    c0, c1 = int(size * 0.20), int(size * 0.80)
    mask[r0:r1, c0:c1] = 1
    mot = mot * mask
    if mot.max() <= 0:
        return int(size*0.60), int(size*0.92), int(size*0.30), int(size*0.70)
    rr, cc = np.unravel_index(int(np.argmax(mot)), mot.shape)
    h = int(size * 0.12); w = int(size * 0.16)
    return max(0, rr-h), min(size, rr+h), max(0, cc-w), min(size, cc+w)


def _mouth_env(frames: list[bytes], size: int) -> np.ndarray:
    """Per-frame mouth-MOTION envelope: frame-to-frame abs diff in the auto-located mouth
    box (captures lip movement, which should track audio energy). Frames are raw RGB."""
    if len(frames) < 3:
        return np.zeros(len(frames), np.float32)
    arr = _frames_arr(frames, size)
    r0, r1, c0, c1 = _auto_mouth_box(arr, size)
    region = arr[:, r0:r1, c0:c1, :]
    # temporal motion: |frame[i]-frame[i-1]|; prepend 0 so length matches frame count
    mot = np.abs(np.diff(region, axis=0)).mean(axis=(1, 2, 3))
    return np.concatenate([[0.0], mot]).astype(np.float32)


def _xcorr_lag(m: np.ndarray, a: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Lag (in frames) where mouth-motion m best matches audio-rms a. Positive lag = mouth
    LAGS the audio (mouth moves `lag` frames after the sound). Returns (lag, corr)."""
    n = min(len(m), len(a))
    if n < 8:
        return 0, 0.0
    m = m[:n].astype(np.float64); a = a[:n].astype(np.float64)
    m = (m - m.mean()) / (m.std() + 1e-9)
    a = (a - a.mean()) / (a.std() + 1e-9)
    best_lag, best = 0, -2.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            mm, aa = m[lag:], a[: n - lag]
        else:
            mm, aa = m[: n + lag], a[-lag:]
        if len(mm) < 8:
            continue
        c = float((mm * aa).mean())
        if c > best:
            best, best_lag = c, lag
    return best_lag, best


# --------------------------------------------------------------------- one clip run
async def _run_clip(ws, pcm: np.ndarray, fps: float, size: int) -> dict:
    voice_s = len(pcm) / 16000.0
    pcm16 = (np.clip(pcm, -1, 1) * 32767).astype(np.int16)
    state = {"frames": [], "vstart": None, "vend": None, "real": 0, "clock_times": []}

    await ws.send(json.dumps({"type": "speech_start"}))
    t_send0 = time.time()
    step = 3200  # 0.2s @16k, real-time paced
    # interleave receive while sending so we don't miss frames
    async def sender():
        for i in range(0, len(pcm16), step):
            await ws.send(pcm16[i:i + step].tobytes())
            await asyncio.sleep(step / 16000.0)
        await ws.send(json.dumps({"type": "speech_end"}))
        return time.time()

    send_task = asyncio.create_task(sender())
    t_speech_end = None
    deadline = time.time() + voice_s + 25
    while time.time() < deadline:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=8)
        except asyncio.TimeoutError:
            break
        if isinstance(msg, (bytes, bytearray)):
            state["frames"].append(bytes(msg))
        else:
            e = json.loads(msg)
            if e["type"] == "video_start":
                state["vstart"] = time.time()
            elif e["type"] == "video_clock":
                state["real"] = int(e.get("frames", state["real"]))
                state["clock_times"].append((time.time(), state["real"]))
            elif e["type"] == "video_end":
                state["vend"] = time.time()
                if send_task.done():
                    break
        if t_speech_end is None and send_task.done():
            t_speech_end = send_task.result()
    if t_speech_end is None:
        try:
            t_speech_end = send_task.result()
        except Exception:  # noqa: BLE001
            t_speech_end = time.time()
        send_task.cancel()

    # distinct rendered frames (drop consecutive byte-identical held/neutral repeats)
    distinct = []
    prev = None
    for f in state["frames"]:
        if f != prev:
            distinct.append(f)
        prev = f
    stride = max(1, round(25 / fps))
    spf = stride * 640
    a_env = _audio_env(pcm, spf)
    m_env = _mouth_env(distinct, size)
    lag, corr = _xcorr_lag(m_env, a_env, max_lag=12)
    voice_frames = voice_s * fps
    render_wait = (state["vstart"] - t_speech_end) if state["vstart"] else None
    return {
        "voice_s": round(voice_s, 2),
        "voice_frames": round(voice_frames, 1),
        "distinct_frames": len(distinct),
        "real_reported": state["real"],
        "tail": round(state["real"] - voice_frames, 1),
        "render_wait_s": round(render_wait, 2) if render_wait is not None else None,
        "lip_lag_frames": lag,
        "lip_lag_s": round(lag / fps, 3),
        "xcorr": round(corr, 3),
    }


# --------------------------------------------------------------------- a config
async def _measure_async(overrides: dict, clips: dict[str, Path]) -> dict:
    fps = float(overrides.get("FPS", 12.5))
    async with websockets.connect(WS, max_size=None, ping_interval=None) as ws:
        await ws.send(json.dumps({"type": "config", "fps": fps}))
        size = int(overrides.get("SIZE", 512))
        per = {}
        for name, path in clips.items():
            pcm = _load_16k(path)
            per[name] = await _run_clip(ws, pcm, fps, size)
            await asyncio.sleep(0.6)
    return per


def measure(overrides: dict) -> dict:
    clips = {n: CLIP_DIR / f"{n}.wav" for n, *_ in CLIP_SPEC}
    if not all(p.exists() for p in clips.values()):
        make_clips()
    _kill_server()
    proc = _start_server(overrides)
    try:
        health = _wait_health()
        size = int(health.get("size", overrides.get("SIZE", 512)))
        overrides = {**overrides, "SIZE": size}
        per = asyncio.run(_measure_async(overrides, clips))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill()
    # aggregate lip lag over medium+long (robust), render_wait over short
    lags = [per[k]["lip_lag_s"] for k in ("medium", "long") if k in per]
    xcs = [per[k]["xcorr"] for k in ("medium", "long") if k in per]
    result = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": overrides,
        "lip_lag_s_med": round(float(np.median(lags)), 3) if lags else None,
        "xcorr_med": round(float(np.median(xcs)), 3) if xcs else None,
        "render_wait_short": per.get("short", {}).get("render_wait_s"),
        "tail_long": per.get("long", {}).get("tail"),
        "per_clip": per,
    }
    JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")
    return result


async def _collect_frames(fps: float, size: int, clip: Path) -> list[bytes]:
    pcm = _load_16k(clip)
    pcm16 = (np.clip(pcm, -1, 1) * 32767).astype(np.int16)
    frames: list[bytes] = []
    async with websockets.connect(WS, max_size=None, ping_interval=None) as ws:
        await ws.send(json.dumps({"type": "config", "fps": fps}))
        await ws.send(json.dumps({"type": "speech_start"}))

        async def sender():
            step = 3200
            for i in range(0, len(pcm16), step):
                await ws.send(pcm16[i:i + step].tobytes())
                await asyncio.sleep(step / 16000.0)
            await ws.send(json.dumps({"type": "speech_end"}))

        st = asyncio.create_task(sender())
        deadline = time.time() + len(pcm16) / 16000.0 + 20
        ended = False
        while time.time() < deadline:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=8)
            except asyncio.TimeoutError:
                break
            if isinstance(msg, (bytes, bytearray)):
                frames.append(bytes(msg))
            elif json.loads(msg)["type"] == "video_end" and st.done():
                ended = True
                break
        if not ended:
            st.cancel()
    # distinct (drop held/neutral repeats)
    out, prev = [], None
    for f in frames:
        if f != prev:
            out.append(f)
        prev = f
    return out


def diag(overrides: dict) -> None:
    import imageio.v2 as imageio

    fps = float(overrides.get("FPS", 12.5))
    clip = CLIP_DIR / "long.wav"
    if not clip.exists():
        make_clips()
    _kill_server()
    proc = _start_server(overrides)
    try:
        health = _wait_health()
        size = int(health.get("size", 512))
        frames = asyncio.run(_collect_frames(fps, size, clip))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill()

    d = ROOT / "output" / "tune_diag"
    d.mkdir(parents=True, exist_ok=True)
    arr = _frames_arr(frames, size)
    r0, r1, c0, c1 = _auto_mouth_box(arr, size)
    print(f"distinct frames={len(frames)} size={size} auto_mouth_box=rows[{r0}:{r1}] cols[{c0}:{c1}]")

    mid = frames[len(frames) // 2]
    mid_img = np.frombuffer(mid, np.uint8).reshape(size, size, 3).copy()
    rest = np.median(arr, axis=0).astype(np.uint8)
    # motion heatmap (per-pixel temporal motion), normalized to 0-255 grayscale->RGB
    mot = np.abs(np.diff(arr, axis=0)).mean(axis=(0, 3))
    hm = (255 * mot / (mot.max() + 1e-9)).astype(np.uint8)
    hm_rgb = np.stack([hm, hm, hm], axis=-1)
    # draw the box on the mid frame
    mid_img[r0:r1, c0:c0+2] = [255, 0, 0]; mid_img[r0:r1, c1-2:c1] = [255, 0, 0]
    mid_img[r0:r0+2, c0:c1] = [255, 0, 0]; mid_img[r1-2:r1, c0:c1] = [255, 0, 0]
    imageio.imwrite(d / "mid_with_box.png", mid_img)
    imageio.imwrite(d / "rest.png", rest)
    imageio.imwrite(d / "motion_heatmap.png", hm_rgb)

    # envelopes: openness (diff vs rest) and motion (temporal diff), both on the box
    region = arr[:, r0:r1, c0:c1, :]
    openness = np.abs(region - np.median(region, axis=0, keepdims=True)).mean(axis=(1, 2, 3))
    motion = np.concatenate([[0.0], np.abs(np.diff(region, axis=0)).mean(axis=(1, 2, 3))])
    pcm = _load_16k(clip)
    spf = max(1, round(25 / fps)) * 640
    a = _audio_env(pcm, spf)
    for label, env in (("openness", openness), ("motion", motion)):
        lag, corr = _xcorr_lag(env, a, 12)
        print(f"  {label:9s}: best_lag={lag:+d} frames ({lag/fps:+.3f}s) xcorr={corr:+.3f}")
    (d / "envelopes.json").write_text(json.dumps({
        "fps": fps, "openness": openness.tolist(), "motion": motion.tolist(),
        "audio_rms": a.tolist(),
    }))
    print(f"saved diagnostics -> {d}")


async def _collect_after_vstart(fps: float, clip: Path) -> list[bytes]:
    """Collect frames received AFTER video_start, in order, NO dedup -- the pump emits one
    frame per output period, so frame i maps to clip-render-time i/fps (a uniform timeline
    suitable for absolute A/V-offset measurement)."""
    pcm = _load_16k(clip)
    pcm16 = (np.clip(pcm, -1, 1) * 32767).astype(np.int16)
    frames: list[bytes] = []
    started = False
    async with websockets.connect(WS, max_size=None, ping_interval=None) as ws:
        await ws.send(json.dumps({"type": "config", "fps": fps}))
        await ws.send(json.dumps({"type": "speech_start"}))

        async def sender():
            step = 3200
            for i in range(0, len(pcm16), step):
                await ws.send(pcm16[i:i + step].tobytes())
                await asyncio.sleep(step / 16000.0)
            await ws.send(json.dumps({"type": "speech_end"}))

        st = asyncio.create_task(sender())
        deadline = time.time() + len(pcm16) / 16000.0 + 25
        while time.time() < deadline:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=8)
            except asyncio.TimeoutError:
                break
            if isinstance(msg, (bytes, bytearray)):
                if started:
                    frames.append(bytes(msg))
            else:
                e = json.loads(msg)
                if e["type"] == "video_start":
                    started = True
                elif e["type"] == "video_end" and st.done():
                    break
        if not st.done():
            st.cancel()
    return frames


def _openness_env(frames: list[bytes], size: int) -> np.ndarray:
    arr = _frames_arr(frames, size)
    r0, r1, c0, c1 = _auto_mouth_box(arr, size)
    region = arr[:, r0:r1, c0:c1, :]
    rest = np.median(region, axis=0, keepdims=True)
    return np.abs(region - rest).mean(axis=(1, 2, 3)).astype(np.float32)


def calib(overrides: dict) -> dict:
    """Measure absolute lip A/V offset with the calibration clip (sharp silence->burst
    onsets). Two independent estimates: xcorr lag, and per-syllable onset delay."""
    clip = CLIP_DIR / "calib.wav"
    if not clip.exists():
        make_clips()
    onsets = json.loads((CLIP_DIR / "calib_onsets.json").read_text())
    fps = float(overrides.get("FPS", 12.5))
    _kill_server()
    proc = _start_server(overrides)
    try:
        health = _wait_health()
        size = int(health.get("size", 512))
        frames = asyncio.run(_collect_after_vstart(fps, clip))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill()

    op = _openness_env(frames, size)
    pcm = _load_16k(clip)
    spf = max(1, round(25 / fps)) * 640
    a = _audio_env(pcm, spf)
    lag_x, corr = _xcorr_lag(op, a, 12)

    # per-syllable onset delay: mouth-open rise vs known audio onset. Per-burst LOCAL
    # baseline (the settled silence just before it) so a global threshold can't mis-fire.
    peak = float(np.percentile(op, 95))
    delays = []
    for ot in onsets:
        of = int(round(ot * fps))
        lo = max(0, of - 8)
        local_base = float(np.median(op[lo:max(lo + 1, of - 1)])) if of - 1 > lo else float(op[lo])
        thr = local_base + 0.35 * (peak - local_base)
        win = range(max(1, of - 2), min(len(op), of + 12))
        hit = next((i for i in win if op[i] >= thr), None)
        if hit is not None:
            delays.append(hit - of)
    onset_lag = float(np.median(delays)) if delays else None
    result = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "config": {**overrides, "SIZE": size},
        "method": "calib", "n_frames": len(frames),
        "xcorr_lag_frames": lag_x, "xcorr_lag_s": round(lag_x / fps, 3), "xcorr": round(corr, 3),
        "onset_lag_frames": onset_lag,
        "onset_lag_s": round(onset_lag / fps, 3) if onset_lag is not None else None,
        "onset_delays": delays,
    }
    with open(JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result, indent=2))
    return result


def align(overrides: dict, clip_name: str = "vlong") -> dict:
    """Robust intrinsic A/V-offset measurement on CONTINUOUS speech (Ditto animates
    continuous speech far more reliably than isolated syllables). No-dedup post-video_start
    frames (clean i/fps time base) + smoothed openness vs smoothed audio xcorr, with the
    full per-lag curve so the peak's clarity is visible."""
    clip = CLIP_DIR / f"{clip_name}.wav"
    if not clip.exists():
        make_clips()
    fps = float(overrides.get("FPS", 12.5))
    _kill_server()
    proc = _start_server(overrides)
    try:
        health = _wait_health()
        size = int(health.get("size", 512))
        frames = asyncio.run(_collect_after_vstart(fps, clip))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill()

    op = _openness_env(frames, size)
    pcm = _load_16k(clip)
    spf = max(1, round(25 / fps)) * 640
    a = _audio_env(pcm, spf)

    def sm(x, k=3):
        return np.convolve(x, np.ones(k) / k, mode="same")

    def z(x):
        return (x - x.mean()) / (x.std() + 1e-9)

    n = min(len(op), len(a))
    zo, za = z(sm(op[:n])), z(sm(a[:n]))
    curve = {}
    best_lag, best = 0, -2.0
    for lag in range(-10, 11):
        if lag >= 0:
            mm, aa = zo[lag:], za[: n - lag]
        else:
            mm, aa = zo[: n + lag], za[-lag:]
        c = float((mm * aa).mean()) if len(mm) > 8 else 0.0
        curve[lag] = round(c, 3)
        if c > best:
            best, best_lag = c, lag
    print(f"n_frames={len(frames)} matched={n} size={size}")
    print("lag(frames):corr  (negative lag = mouth LEADS audio)")
    for lag in range(-10, 11):
        bar = "#" * int(max(0, curve[lag]) * 40)
        mark = "  <== peak" if lag == best_lag else ""
        print(f"  {lag:+3d}: {curve[lag]:+.3f} {bar}{mark}")
    print(f"\nBEST lag={best_lag:+d} frames = {best_lag/fps:+.3f}s (corr {best:+.3f})")
    print(f"  => mouth {'LEADS' if best_lag < 0 else 'LAGS'} the voice by {abs(best_lag)/fps:.2f}s")
    print(f"  => suggested DITTO_SYNC_LEAD_S adjust: {'+' if best_lag<0 else '-'}{abs(best_lag)/fps:.2f}s "
          f"(release audio {'earlier' if best_lag<0 else 'later'} to match)")
    result = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "config": {**overrides, "SIZE": size},
              "method": "align", "clip": clip_name, "best_lag_frames": best_lag,
              "best_lag_s": round(best_lag / fps, 3), "best_corr": round(best, 3), "curve": curve}
    with open(JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")
    return result


async def _drive_multi_async(fps: float, size: int, feedx: float = 1.0) -> list[dict]:
    """Drive multi.wav PER sentence (speech_start -> real-time PCM -> speech_end -> gap),
    timestamping every frame + marker, then segment into per-sentence metrics. Models the
    real pipeline (one speech segment per sentence) so we see per-sentence start-latency and
    whether the video covers the whole sentence (no trim)."""
    pcm = _load_16k(CLIP_DIR / "multi.wav")
    spans = json.loads((CLIP_DIR / "multi_spans.json").read_text())
    ev: list[tuple] = []  # (t, kind, val)

    async with websockets.connect(WS, max_size=None, ping_interval=None) as ws:
        await ws.send(json.dumps({"type": "config", "fps": fps}))

        async def receiver():
            try:
                async for m in ws:
                    if isinstance(m, (bytes, bytearray)):
                        ev.append((time.time(), "frame", None))
                    else:
                        e = json.loads(m)
                        ev.append((time.time(), e["type"], e.get("frames")))
            except Exception:  # noqa: BLE001
                pass

        rcv = asyncio.create_task(receiver())
        bounds = []  # (sent_idx, t_speech_start)
        for i, (s0, dur) in enumerate(spans):
            seg = pcm[int(s0 * 16000): int((s0 + dur) * 16000)]
            seg16 = (np.clip(seg, -1, 1) * 32767).astype(np.int16)
            t_ss = time.time()
            bounds.append((t_ss, dur))
            await ws.send(json.dumps({"type": "speech_start"}))
            step = 3200
            for j in range(0, len(seg16), step):
                await ws.send(seg16[j:j + step].tobytes())
                # feedx>1 models real TTS, which streams faster than real time -> the
                # diffuser gets its warmup features sooner -> realistic start latency.
                await asyncio.sleep(step / 16000.0 / feedx)
            await ws.send(json.dumps({"type": "speech_end"}))
            await asyncio.sleep(4.0)  # gap + let this sentence finish rendering/playing
        await asyncio.sleep(1.0)
        rcv.cancel()

    # segment events by sentence (each sentence spans [t_ss_i, t_ss_{i+1}))
    starts = [b[0] for b in bounds] + [float("inf")]
    out = []
    for i, (t_ss, dur) in enumerate(bounds):
        lo, hi = starts[i], starts[i + 1]
        seg_ev = [e for e in ev if lo <= e[0] < hi]
        vstart = next((t for t, k, v in seg_ev if k == "video_start"), None)
        vend = next((t for t, k, v in seg_ev if k == "video_end"), None)
        clocks = [v for t, k, v in seg_ev if k == "video_clock" and v is not None]
        real_frames = max(clocks) if clocks else 0
        voice_frames = dur * fps
        play_span = (vend - vstart) if (vstart and vend) else None
        out.append({
            "sentence": i, "voice_s": round(dur, 2), "voice_frames": round(voice_frames, 1),
            "start_latency_s": round(vstart - t_ss, 2) if vstart else None,
            "real_frames": real_frames,
            "coverage": round(real_frames / voice_frames, 2) if voice_frames else None,
            "silent_tail_s": round(max(0, real_frames - voice_frames) / fps, 2),
            "play_rate": round(real_frames / play_span, 1) if play_span and play_span > 0 else None,
        })
    return out


def live(overrides: dict) -> dict:
    fps = float(overrides.get("FPS", 12.5))
    feedx = float(overrides.pop("FEEDX", 1.0))  # harness-only: TTS feed-rate multiplier
    # Default to live, but honor an explicit PRERENDER so this multi-sentence driver can
    # also reproduce compute-first behavior (the inter-sentence truncation lives there).
    overrides = {"PRERENDER": "0", **overrides}
    if not (CLIP_DIR / "multi.wav").exists():
        make_clips()
    _kill_server()
    proc = _start_server(overrides)
    try:
        health = _wait_health()
        size = int(health.get("size", 512))
        per = asyncio.run(_drive_multi_async(fps, size, feedx))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill()

    cov = [s["coverage"] for s in per if s["coverage"] is not None]
    lat = [s["start_latency_s"] for s in per if s["start_latency_s"] is not None]
    tail = [s["silent_tail_s"] for s in per]
    rate = [s["play_rate"] for s in per if s["play_rate"] is not None]
    result = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "config": {**overrides, "SIZE": size},
        "method": "live",
        "min_coverage": min(cov) if cov else None,
        "max_start_latency_s": max(lat) if lat else None,
        "max_silent_tail_s": max(tail) if tail else None,
        "min_play_rate": min(rate) if rate else None, "fps": fps,
        "per_sentence": per,
    }
    with open(JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_sentence"}, indent=2))
    print("--- per sentence ---")
    for s in per:
        print(f"  s{s['sentence']} voice={s['voice_s']}s start_lat={s['start_latency_s']}s "
              f"coverage={s['coverage']} tail={s['silent_tail_s']}s play_rate={s['play_rate']} "
              f"(real {s['real_frames']}/{s['voice_frames']})")
    return result


def _parse_set(s: str) -> dict:
    out = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split("=", 1)
        out[k.strip().upper()] = v.strip()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("clips")
    mp = sub.add_parser("measure")
    mp.add_argument("--set", default="", help="OVERLAP=10,FPS=12.5,PRERENDER=1,STEPS=25,SIZE=512")
    dp = sub.add_parser("diag")
    dp.add_argument("--set", default="")
    cp = sub.add_parser("calib")
    cp.add_argument("--set", default="")
    alp = sub.add_parser("align")
    alp.add_argument("--set", default="")
    alp.add_argument("--clip", default="vlong")
    lp = sub.add_parser("live")
    lp.add_argument("--set", default="")
    args = ap.parse_args()

    if args.cmd == "clips":
        make_clips()
    elif args.cmd == "diag":
        diag(_parse_set(args.set))
    elif args.cmd == "calib":
        calib(_parse_set(args.set))
    elif args.cmd == "align":
        align(_parse_set(args.set), args.clip)
    elif args.cmd == "live":
        live(_parse_set(args.set))
    elif args.cmd == "measure":
        ov = _parse_set(args.set)
        r = measure(ov)
        print(json.dumps({k: v for k, v in r.items() if k != "per_clip"}, indent=2))
        print("--- per clip ---")
        for name, m in r["per_clip"].items():
            print(f"  {name:7s} lip_lag={m['lip_lag_s']:+.3f}s (frames {m['lip_lag_frames']:+d}, "
                  f"xcorr {m['xcorr']:+.2f}) render_wait={m['render_wait_s']}s tail={m['tail']} "
                  f"distinct={m['distinct_frames']}")


if __name__ == "__main__":
    main()
