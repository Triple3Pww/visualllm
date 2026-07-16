"""Turn drivers for the latency harness.

Two ways to drive a real turn through the live pipeline:
  * run_probe  -- a headless aiortc client (like a browser) that plays a wav as the mic and
    records the bot's A/V. Precise play-start clock -> a precise pre-t0 capture anchor, and the
    received-audio arrival (E, transport). F (browser jitter/decode/playout) stays estimated.
  * run_browser_turns -- a REAL headless Chromium (Playwright) with the wav as a fake mic, on
    /studio/?measure=1. Its WebAudio+getStats beacon lands the REAL browser output delay
    (E + F) in pipeline.log as [client-playout] lines. Falls back to the probe if Playwright is
    unavailable. The looping fake mic gives an approximate play-start, so capture is left unknown
    on this path (read it from a --no-browser probe run).
"""
from __future__ import annotations

import asyncio
import time
import wave
from pathlib import Path

import aiohttp
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer, MediaRecorder, MediaRelay

# Reuse the probe's wav builder + lip-offset analyser (single source of truth).
from scripts._webrtc_probe import build_mic_wav, lip_offset_from_mp4, wait_ice

ROOT = Path(__file__).resolve().parent.parent.parent
OFFER_URL = "http://127.0.0.1:7860/api/offer"
MP4 = str(ROOT / "output" / "measure_live.mp4")


def _audio_rms(frame):
    """RMS of one decoded aiortc AudioFrame (int16 PCM) -> float."""
    s = frame.to_ndarray()
    if s.size == 0:
        return 0.0
    s = s.astype(np.float64)
    return float(np.sqrt(np.mean(s * s)))


def speech_duration(mic_wav):
    with wave.open(mic_wav, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def speech_end_epoch(mic_wav, lead, play_start_epoch):
    """When the user's audio goes silent, in epoch seconds: playback starts at play_start_epoch
    and the wav is [lead silence | speech | tail silence]."""
    return play_start_epoch + lead + speech_duration(mic_wav)


# ----------------------------------------------------------------- headless aiortc probe
async def run_probe(mic_wav: str, lead: float, tail: float, duration: float):
    """Connect like a browser, play the mic wav, record + time the bot's A/V.
    Returns (vwall, awall, connect_t): video-arrival wall times, (arrival_epoch, rms) per audio
    frame, and the connect epoch."""
    vwall: list[float] = []
    awall: list[tuple[float, float]] = []
    mic = build_mic_wav(mic_wav, lead, tail)

    pc = RTCPeerConnection()
    pc.addTrack(MediaPlayer(mic).audio)
    pc.addTransceiver("video", direction="recvonly")
    tracks: dict = {}
    pc.on("track", lambda t: tracks.__setitem__(t.kind, t))

    await pc.setLocalDescription(await pc.createOffer())
    await wait_ice(pc)
    connect_t = time.time()
    async with aiohttp.ClientSession() as s:
        async with s.post(OFFER_URL, json={"sdp": pc.localDescription.sdp,
                                           "type": "offer"}) as r:
            ans = await r.json()
    await pc.setRemoteDescription(RTCSessionDescription(sdp=ans["sdp"], type=ans["type"]))

    for _ in range(50):
        if "video" in tracks and "audio" in tracks:
            break
        await asyncio.sleep(0.1)

    async def vpump(track):
        while True:
            try:
                await track.recv()
            except Exception:
                return
            vwall.append(time.time())

    async def apump(track):
        while True:
            try:
                frame = await track.recv()
            except Exception:
                return
            awall.append((time.time(), _audio_rms(frame)))

    relay = MediaRelay()
    recorder = MediaRecorder(MP4)
    if "video" in tracks:
        recorder.addTrack(relay.subscribe(tracks["video"]))
        asyncio.ensure_future(vpump(relay.subscribe(tracks["video"])))
    if "audio" in tracks:
        recorder.addTrack(relay.subscribe(tracks["audio"]))
        asyncio.ensure_future(apump(relay.subscribe(tracks["audio"])))
    await recorder.start()
    print(f"  connected (pc_id={ans.get('pc_id')}, tracks={list(tracks)}); capturing {duration}s...")
    await asyncio.sleep(duration)
    await recorder.stop()
    await pc.close()
    return vwall, awall, connect_t


def probe_metrics(vwall, awall, connect_t, fps):
    m = {"video_frames": len(vwall), "audio_packets": len(awall)}
    if len(vwall) >= 5:
        w = np.array(vwall)
        gaps = np.diff(w)
        m.update(
            startup_s=round(w[0] - connect_t, 2),
            recv_fps=round(len(vwall) / (w[-1] - w[0] + 1e-9), 1),
            frame_ms_mean=round(gaps.mean() * 1000, 1),
            frame_ms_p95=round(float(np.percentile(gaps, 95)) * 1000, 1),
            frame_ms_max=round(gaps.max() * 1000, 1),
            freeze_ms=round(gaps.max() * 1000),
        )
    if len(awall) > 2:
        ag = np.diff(np.array([t for t, _ in awall]))
        m["audio_gap_p95_ms"] = round(float(np.percentile(ag, 95)) * 1000, 1)
        m["audio_gap_max_ms"] = round(ag.max() * 1000, 1)
    off, corr, err = lip_offset_from_mp4(MP4, fps)
    if err:
        m["lip_offset"] = None
        m["lip_offset_note"] = err
    else:
        m["lip_offset_ms"] = round(off * 1000)
        m["lip_offset_corr"] = round(corr, 2)
    return m


# ----------------------------------------------------------------- real Chromium (Playwright)
async def run_browser_turns(mic_wav, n_turns, lead=2.0, tail=6.0):
    """Real headless Chromium: the wav loops as a fake mic, /studio/?measure=1 runs the playout
    beacon, and each loop is one VAD-driven turn (-> a [TTFO] line + a [client-playout] beacon).
    Returns False (caller falls back to run_probe) if Playwright/Chromium is unavailable."""
    try:
        from playwright.async_api import async_playwright
    except Exception:
        print("  [browser] Playwright not installed -- falling back to the headless probe.")
        return False
    # Chrome loops --use-file-for-fake-audio-capture; a [lead | speech | tail] wav thus yields one
    # turn per (lead+speech+tail) seconds. build_mic_wav emits the 16-bit PCM Chrome's fake device
    # wants. %noloop is NOT appended, so it repeats for n_turns.
    driven = build_mic_wav(mic_wav, lead, tail)
    period = lead + speech_duration(mic_wav) + tail
    args = ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
            "--autoplay-policy=no-user-gesture-required",
            f"--use-file-for-fake-audio-capture={driven}"]
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=args)
            page = await browser.new_page()
            await page.goto("http://localhost:7860/studio/?measure=1")
            # Connect (grants the fake mic). #connectBtn is the primary CTA; #micBtn also connects.
            try:
                await page.click("#connectBtn", timeout=6000)
            except Exception:
                await page.click("#micBtn", timeout=6000)
            wait_s = n_turns * period + 4.0
            print(f"  [browser] driving {n_turns} looped turns (~{period:.0f}s each, {wait_s:.0f}s)...")
            await page.wait_for_timeout(int(wait_s * 1000))
            await browser.close()
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [browser] driver error ({e!r}); falling back to the headless probe.")
        return False
