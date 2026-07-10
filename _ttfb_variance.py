"""
First-chunk TTFB *variance* probe for the CUDA-graphs diagnostic (2026-07-05).

benchmark.py measures TOTAL synth time; this measures time-to-first-PCM-byte over
/tts/stream -- the number TTFO actually cares about -- and reports median/max/stddev,
not just an average. Drives a set of opener lengths, EACH REPEATED across N rounds,
so a capture/JIT spike on a novel shape shows up as a round-1 (cold) penalty that
vanishes in rounds 2+ (warm). That first-vs-repeat gap is the whole A-vs-B test:

  - Round-1 spikes that DISAPPEAR on repeats  -> mechanism (A) capture-on-novel-shape
    (fixable by warming the opener band at boot).
  - Tight even on round 1 (isolated, no MuseTalk) -> live jank was (B) contention.

MUST hit the WSL IP directly (localhost relay buffers the stream and fakes TTFB).
Run from Windows system python:  python _ttfb_variance.py --host 172.24.44.238
"""
import argparse
import json
import statistics
import time
import urllib.request

# Opener set: varied lengths spanning the COSYVOICE_FIRST_PIECE band (distinct token
# counts = distinct prefill shapes) + zh comma-split openers. Each is a plausible
# first CLAUSE the pipeline would send.
OPENERS = [
    ("en08", "Sure, let me check that."),
    ("en12", "Good morning, the weather looks clear today."),
    ("en16", "That's a great question, and here is what I found for you today."),
    ("en20", "Absolutely, I can help with that, so let me walk you through the details step by step."),
    ("en26", "Well, to give you an accurate answer on that, I first want to make sure I understand exactly what you are asking about here."),
    ("zh06", "你好，讓我看一下。"),
    ("zh10", "早安，今天台北天氣晴朗。"),
    ("zh16", "這是一個很好的問題，讓我為您說明一下今天的情況。"),
]


def ttfb(host, text, sr=16000, speed=1.0):
    """POST /tts/stream, return seconds until the first PCM byte arrives."""
    body = json.dumps({"text": text, "sample_rate": sr, "speed": speed}).encode()
    req = urllib.request.Request(
        f"http://{host}:8001/tts/stream", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:
        while True:
            chunk = resp.read(1)  # first byte = first audio out
            if chunk:
                return time.perf_counter() - t0
            # empty read w/o EOF is unusual; loop guards against 0-length keep-alives


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="172.24.44.238", help="WSL IP (NOT localhost)")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--tag", default="", help="label for this config in the output")
    args = ap.parse_args()

    print(f"host={args.host} rounds={args.rounds} tag={args.tag or '(none)'}")
    # one throwaway to shake off connection setup jitter (NOT counted)
    try:
        ttfb(args.host, "warm the socket")
    except Exception as e:
        print(f"server not reachable: {e!r}")
        return

    per_opener = {name: [] for name, _ in OPENERS}
    round1, later = [], []
    for r in range(1, args.rounds + 1):
        for name, text in OPENERS:
            t = ttfb(args.host, text)
            per_opener[name].append(t)
            (round1 if r == 1 else later).append(t)
            print(f"  r{r} {name}: {t:.3f}s")

    print("\nper-opener (round1 | rounds2+ median):")
    for name, _ in OPENERS:
        ts = per_opener[name]
        r1 = ts[0]
        rest = statistics.median(ts[1:]) if len(ts) > 1 else float("nan")
        flag = "  <-- COLD SPIKE" if (len(ts) > 1 and r1 > rest * 1.5) else ""
        print(f"  {name}: r1={r1:.3f}  rest_med={rest:.3f}{flag}")

    allt = round1 + later
    def stats(xs):
        return (statistics.median(xs), max(xs),
                statistics.pstdev(xs) if len(xs) > 1 else 0.0)
    am, ax, asd = stats(allt)
    print(f"\nALL   n={len(allt)}  median={am:.3f}  max={ax:.3f}  stddev={asd:.3f}")
    if later:
        r1m, r1x, r1sd = stats(round1)
        lm, lx, lsd = stats(later)
        print(f"round1 (cold)  median={r1m:.3f}  max={r1x:.3f}  stddev={r1sd:.3f}")
        print(f"rounds2+ (warm) median={lm:.3f}  max={lx:.3f}  stddev={lsd:.3f}")
        print(f"COLD PENALTY (round1_med - warm_med) = {r1m - lm:+.3f}s")


if __name__ == "__main__":
    main()
