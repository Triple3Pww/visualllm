"""Stacked per-stage latency waterfall (median deltas), zh vs en.

Run: python paper/scripts/make_waterfall.py

Note: segments are per-stage MEDIANS, medianed independently, so a bar's total is the
sum of stage medians -- close to, but not identical with, the median TTFO (Table 3).
The annotation says so explicitly.
"""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parent.parent / "data"
FIGS = Path(__file__).resolve().parent.parent / "figures"

STAGES = [
    ("LLM -> TTS (sentence-1 flush)", "LLM (overlaps probe speech; see text)"),
    ("TTS synth first chunk", "TTS first chunk"),
    ("TTS -> bot-start (steady lead-hold)", "Sync lead-hold"),
    ("Transport + encode + network", "Transport + network"),
    ("Browser jitter + decode + playout", "Jitter + playout (est.)"),
]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def medians(lang):
    runs = [json.loads(f.read_text(encoding="utf-8"))
            for f in sorted(DATA.glob(f"{lang}_run_*.json"))]
    out = []
    for key, _ in STAGES:
        ds = [row["delta"] for r in runs for row in r.get("waterfall", [])
              if row["stage"] == key and row["delta"] is not None]
        out.append(statistics.median(ds) if ds else 0.0)
    return out


def main():
    langs = [l for l in ("zh", "en") if list(DATA.glob(f"{l}_run_*.json"))]
    fig, ax = plt.subplots(figsize=(7.0, 1.4 + 0.9 * len(langs)))
    for yi, lang in enumerate(langs):
        deltas = medians(lang)
        left = 0.0
        for (key, label), d, c in zip(STAGES, deltas, COLORS):
            ax.barh(yi, d, left=left, color=c, edgecolor="white",
                    label=label if yi == 0 else None)
            left += d
        ax.annotate(f"Σ stage medians {left:.2f}s", (left, yi), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9)
    ax.set_yticks(range(len(langs)), [l.upper() for l in langs])
    ax.set_xlabel("seconds after the user stops speaking\n"
                  "(per-stage medians of 10 fresh sessions; TTFO medians in Table 3)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncols=3, fontsize=8,
              frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"waterfall.{ext}", dpi=300)
    print("wrote", FIGS / "waterfall.png")


if __name__ == "__main__":
    main()
