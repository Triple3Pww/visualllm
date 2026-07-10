"""One arm of the CosyVoice2-vs-3 A/B: measure the LIVE :8001 server, append a tagged record.

The operator brings the arm up (COSYVOICE_MODEL_DIR selects v2 or v3) and runs this once per
cycle. Sequential arms mean thermal/background drift is a confound, so we run A,B,A,B and
compare medians WITHIN a cycle -- never a single A against a single B.

This driver deliberately does NOT start or stop servers: Windows process tools hang for tens of
seconds under CPU load on this box, and the whole experiment is a timing measurement.

  python _ab_run.py --host <WSL_IP> --tag v2 --cycle 1
"""
import argparse
import json
import pathlib
import statistics
import time

from _ttfb_variance import OPENERS, ttfb
from _zh_audio_ab import ZH, analyze, synth

OUT = pathlib.Path(__file__).parent / "output" / "cv3_ab.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="WSL IP, NOT localhost")
    ap.add_argument("--tag", required=True, choices=["v2", "v3"])
    ap.add_argument("--cycle", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--zh-runs", type=int, default=5)
    args = ap.parse_args()

    ttfb(args.host, "warm the socket")  # not counted

    ttfbs = []
    for r in range(args.rounds):
        for name, text in OPENERS:
            t = ttfb(args.host, text)
            ttfbs.append(t)
            print(f"  c{args.cycle} {args.tag} r{r+1} {name}: {t:.3f}s")

    zh = []
    for i in range(args.zh_runs):
        dur, longest, _frac, lead = analyze(synth(args.host, ZH))
        zh.append({"dur": dur, "longest_sil": longest, "lead": lead})
        print(f"  c{args.cycle} {args.tag} zh{i+1}: dur={dur:.2f}s lead={lead:.2f}s")

    rec = {"tag": args.tag, "cycle": args.cycle, "ttfb": ttfbs, "zh": zh, "ts": time.time()}
    OUT.parent.mkdir(exist_ok=True)
    records = json.loads(OUT.read_text()) if OUT.exists() else []
    records.append(rec)
    OUT.write_text(json.dumps(records, indent=2))

    print(f"\n{args.tag} cycle{args.cycle}  n={len(ttfbs)}  "
          f"median={statistics.median(ttfbs):.3f}  max={max(ttfbs):.3f}  "
          f"stddev={statistics.pstdev(ttfbs):.3f}")
    print(f"  zh lead median={statistics.median(x['lead'] for x in zh):.2f}s")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
