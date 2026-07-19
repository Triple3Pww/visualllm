"""Barge-in TTS residue probe (P56, 29th session). Confirms + measures the bug that a
mid-utterance abandon (a live barge-in) leaves CosyVoice generating on the shared GPU, slowing
the NEXT turn's TTS.

WHY IT IS FAITHFUL: it uses aiohttp with the SAME `async with session.post(...)` cancel
semantics as local_services/cosyvoice_tts.py::run_tts, and one persistent session (as the
pipeline does). A urllib `resp.close()` does NOT reliably signal disconnect to the server, so
GeneratorExit mostly does not fire on that path -- do NOT measure this with urllib.

Two things it measures:
  * repro   : short-turn TTFB after a DRAINED long request (clean) vs after an ABANDONED one.
  * discrim : + a short ABANDON arm, to prove the residue SCALES with abandoned generation
              (long-abandon >> short-abandon ~= clean). That is how we know it is leftover
              speech-token generation, not a fixed per-abandon cost.

Run (Windows system python; hit the WSL IP, NOT localhost -- the relay buffers the stream):
  python _bargein_residue_probe.py --host 172.24.44.238

Baseline observed 2026-07-16 (v2, graphs on): clean ~0.46s, long-abandon ~1.38s (+0.9s),
short-abandon ~0.47s (+0.0s).  See docs/BARGEIN-RESIDUE-HANDOFF.md + PROBLEMS-AND-FIXES.md P56.
"""
import argparse
import asyncio
import statistics
import time

import aiohttp

LONG = ("人工智慧是電腦科學的一個領域，專注於建立能夠推理、學習並解決問題的系統，"
        "這個領域近年來發展非常快速，應用範圍也越來越廣泛，從語音辨識到自動駕駛都有，"
        "未來還會深入醫療、教育與金融等各行各業，帶來深遠的影響。")
SHORT_ABANDON = "你好嗎，今天過得好嗎"
MEASURE = "早安"
SR = 16000
CHUNK = int(SR * 2 * 0.02)


async def _drain(s, host, text):
    """Consume the whole stream; return TTFB. What run_tts does on a normal turn."""
    t0 = time.perf_counter()
    first = None
    async with s.post(f"http://{host}:8001/tts/stream", json={"text": text, "sample_rate": SR}) as r:
        async for c in r.content.iter_chunked(CHUNK):
            if c and first is None:
                first = time.perf_counter() - t0
    return first


async def _abandon(s, host, text, consume_s):
    """Consume ~consume_s of audio then EXIT the context early = abandon (barge-in)."""
    got = 0
    async with s.post(f"http://{host}:8001/tts/stream", json={"text": text, "sample_rate": SR}) as r:
        async for c in r.content.iter_chunked(CHUNK):
            got += len(c)
            if got / 2 / SR >= consume_s:
                break


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="172.24.44.238", help="WSL IP (NOT localhost)")
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    async with aiohttp.ClientSession() as s:
        await _drain(s, args.host, "warm")  # not counted
        clean, long_ab, short_ab = [], [], []
        for _ in range(args.n):
            await _drain(s, args.host, LONG)
            clean.append(await _drain(s, args.host, MEASURE))
            await _abandon(s, args.host, LONG, consume_s=0.4)
            long_ab.append(await _drain(s, args.host, MEASURE))
            await _abandon(s, args.host, SHORT_ABANDON, consume_s=0.1)
            short_ab.append(await _drain(s, args.host, MEASURE))

    def med(v):
        return statistics.median([x for x in v if x is not None]) if any(x is not None for x in v) else float("nan")

    print(f"next-turn TTFB ({MEASURE!r}), n={args.n}:   median")
    print(f"  A  after DRAIN long      {med(clean):7.3f}")
    print(f"  B  after ABANDON long    {med(long_ab):7.3f}   (residue {med(long_ab)-med(clean):+.3f}s)")
    print(f"  C  after ABANDON short   {med(short_ab):7.3f}   (residue {med(short_ab)-med(clean):+.3f}s)")
    print("\n  B >> C ~= A  => residue is leftover GENERATION on the shared GPU (P56).")
    print("  A None/empty => the server broke the next turn (a fix regression -- see P56 attempt #3).")


if __name__ == "__main__":
    asyncio.run(main())
