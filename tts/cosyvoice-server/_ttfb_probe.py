"""Measure CosyVoice /tts/stream time-to-first-byte (first-chunk latency) + total.
Run against the live server:  python _ttfb_probe.py
"""
import json, time, urllib.request, sys

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001") + "/tts/stream"
TEXTS = [
    "Hi there! How can I help you today?",
    "Sure, let me explain that for you.",
    "The weather looks great this afternoon.",
]


def one(text):
    body = json.dumps({"text": text, "voice": "weather", "sample_rate": 16000}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttfb = None
    nbytes = 0
    with urllib.request.urlopen(req) as r:
        while True:
            chunk = r.read(4096)
            if not chunk:
                break
            if ttfb is None:
                ttfb = time.perf_counter() - t0
            nbytes += len(chunk)
    total = time.perf_counter() - t0
    audio_s = nbytes / 2 / 16000  # int16 mono @16k
    return ttfb, total, audio_s


def main():
    print(f"{'text':<40}{'TTFB(s)':>9}{'total(s)':>9}{'audio(s)':>9}")
    for t in TEXTS:
        ttfb, total, audio_s = one(t)
        print(f"{t[:38]:<40}{ttfb:>9.2f}{total:>9.2f}{audio_s:>9.2f}")


if __name__ == "__main__":
    main()
