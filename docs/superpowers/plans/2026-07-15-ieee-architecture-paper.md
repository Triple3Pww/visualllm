# IEEE Architecture Paper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a complete IEEE-conference-format paper draft (Markdown + BibTeX + figures + fresh measurement data) on the VisualLLm whole-system architecture, per the approved spec `docs/superpowers/specs/2026-07-15-ieee-architecture-paper-design.md`.

**Architecture:** A `paper/` directory holds one Markdown draft (`draft.md`), a BibTeX file grown from day one (`references.bib`), generated figures (`figures/`), and the frozen measurement data + tiny analysis scripts (`data/`, `scripts/`). Measurements come exclusively from the existing `scripts.measure` harness; the paper is written section-by-section against that frozen data.

**Tech Stack:** Python 3.11 (system, has matplotlib 3.10), the existing measure harness, PowerShell for campaign loops, pandoc-style `[@key]` citations for a mechanical LaTeX conversion later.

## Global Constraints

- **NEVER `git push` in this plan** — `main` is public; publishing the draft is the user's decision (spec §Constraints).
- **Leave the system as found:** after the en campaign, restore `.env` `LANGUAGE=zh` and restart the pipeline. Do not modify any pipeline/runtime code — `pipeline/metrics.py` and all of `pipeline/`, `local_services/` are read-only for this plan.
- **Close any `/client/` or `/studio/` browser tab before measure runs** — the avatar server is single-client.
- The live stack must be up for Tasks 2–3 (`:8001/health`, `:7860`, `:8002`). If not, start it with `scripts/launch.ps1` (or ask the user to double-click `Run VisualLLm.exe`).
- Use `--mic output/_zh_q_def.wav` for zh — **`output/q_ai.wav` no longer transcribes on Deepgram** (27th-session landmine).
- All prose in English; `.py` files ASCII-safe; commit after every task with the `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer.
- Citations in the draft use pandoc syntax `[@key]` where `key` exists in `paper/references.bib`.
- Word budget: ~4,200 words of prose total (6 IEEE pages minus figures/tables/refs). Per-section budgets are in each drafting task.

---

### Task 1: Scaffold `paper/`

**Files:**
- Create: `paper/draft.md`, `paper/references.bib`, `paper/figures/.gitkeep`, `paper/data/.gitkeep`, `paper/scripts/.gitkeep`, `paper/README.md`

**Interfaces:**
- Produces: the directory layout every later task writes into; the section skeleton (exact headers) that Tasks 7–12 fill.

- [ ] **Step 1: Create the skeleton files**

`paper/draft.md`:

```markdown
---
title: "VisualLLm: A Fully Local Streaming Architecture for Real-Time Multilingual Talking-Head Agents on a Single Consumer GPU"
author: "[AUTHORS — user to supply names, order, affiliations]"
bibliography: references.bib
---

# Abstract

<!-- Task 12 -->

# 1. Introduction

<!-- Task 12 -->

# 2. Related Work

<!-- Task 7 -->

# 3. System Architecture

<!-- Task 8 -->

# 4. Latency Engineering

<!-- Task 9 -->

# 5. Audio–Visual Synchronization

<!-- Task 10 -->

# 6. Evaluation

<!-- Task 11 -->

# 7. Discussion and Lessons Learned

<!-- Task 12 -->

# 8. Conclusion

<!-- Task 12 -->

# References

<!-- generated from references.bib at conversion time -->
```

`paper/references.bib`: empty file (Task 7 fills it).

`paper/README.md`:

```markdown
# IEEE architecture paper (working draft)

- `draft.md` — the paper (Markdown; converts to IEEEtran LaTeX after the prof approves content).
- `references.bib` — BibTeX; draft cites with pandoc `[@key]`.
- `data/` — frozen measurement runs (`{zh,en}_run_NN.json` = copies of `output/measure_report.json`),
  `environment.md` (commit + .env snapshot the numbers were taken on), `summary.md` (generated).
- `scripts/` — `aggregate.py` (data -> summary.md), `make_waterfall.py` (data -> figures/waterfall.*).
- `figures/` — generated + hand-authored SVG figures.

Spec: `docs/superpowers/specs/2026-07-15-ieee-architecture-paper-design.md`.
Do not push without the user's go-ahead (public repo).
```

- [ ] **Step 2: Verify layout**

Run: `Get-ChildItem -Recurse paper | Select-Object FullName`
Expected: the 6 files above.

- [ ] **Step 3: Commit**

```powershell
git add paper && git commit -m "paper: scaffold IEEE architecture paper workspace"
```

---

### Task 2: zh measurement campaign (10 fresh runs)

**Files:**
- Create: `paper/data/zh_run_01.json` … `zh_run_10.json`, `paper/data/environment.md`

**Interfaces:**
- Consumes: the live stack; `python -m scripts.measure` (writes `output/measure_report.json`).
- Produces: `paper/data/zh_run_NN.json` with the schema Tasks 4–5 read: `meta.ttfo` (float, seconds), `waterfall[]` rows `{stage, delta, cum, source, status}` (stage names incl. `"LLM -> TTS (sentence-1 flush)"`, `"TTS synth first chunk"`, `"TTS -> bot-start (steady lead-hold)"`, `"Transport + encode + network"`, `"Browser jitter + decode + playout"`), `raw.anchors.playout`, `raw.probe.{recv_fps,freeze_ms,audio_gap_p95_ms}`.

- [ ] **Step 1: Verify the stack is healthy**

```powershell
$wslip = (wsl hostname -I).Trim().Split(" ")[0]
curl.exe -s "http://$($wslip):8001/health"; curl.exe -s -o NUL -w "%{http_code}`n" http://localhost:7860/client/; netstat -ano | findstr ":8002.*LISTENING"
```

Expected: TTS health JSON, `200`, one LISTENING line. If any fails: start the stack (`powershell -File scripts/launch.ps1`) and re-check. Confirm no browser tab has `/client/` or `/studio/` open.

- [ ] **Step 2: Freeze the environment the numbers describe**

Write `paper/data/environment.md` containing: `git rev-parse HEAD` output, `git status --short` (the branch has known uncommitted files — record them honestly), the date, and the values of these `.env` keys (read them from `.env`, do not guess): `LANGUAGE, TTS_PROVIDER, COSYVOICE_MODEL, OPENROUTER_MODEL, OPENROUTER_PROVIDER_ONLY, MUSETALK_SYNC_MODE, MUSETALK_FPS, MUSETALK_SIZE, MUSETALK_TRT, MUSETALK_LEAD_FRAMES, COSYVOICE_FIRST_PIECE, COSYVOICE_FIRST_PIECE_ZH, FILLER_WORDS, MUSETALK_SPLIT`. Plus GPU: `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader`.

- [ ] **Step 3: Run the campaign**

```powershell
1..10 | ForEach-Object {
  $n = "{0:d2}" -f $_
  python -m scripts.measure --mic output/_zh_q_def.wav
  Copy-Item output/measure_report.json "paper/data/zh_run_$n.json"
  Start-Sleep -Seconds 5
}
```

(~1 min/run. Each probe connection is a fresh session — the single-client server drops the previous one — which is what the spec requires given the session-degradation bug.)

- [ ] **Step 4: Staleness + outlier sanity check**

```powershell
python -c "import json,glob; [print(f, (d:=json.load(open(f,encoding='utf-8')))['meta']['when'], d['meta']['ttfo']) for f in sorted(glob.glob('paper/data/zh_run_*.json'))]"
```

Expected: 10 rows, `when` strictly increasing and all from today, TTFO values in a plausible 2.3–5 s band and **not identical across consecutive runs** (identical values = the harness reported a stale turn; re-run that index). Known outlier cause: Groq congestion (~+2–3 s on the LLM row) — keep such runs (p95 is supposed to see them) but note them in `environment.md`.

- [ ] **Step 5: Commit**

```powershell
git add paper/data && git commit -m "paper: zh TTFO campaign, 10 fresh-session runs + environment freeze"
```

---

### Task 3: en measurement campaign (10 fresh runs)

**Files:**
- Create: `paper/data/en_run_01.json` … `en_run_10.json`; possibly `output/_en_q_new.wav`
- Modify: `.env` (`LANGUAGE=en`, then **restored to `zh`**); append to `paper/data/environment.md`

**Interfaces:**
- Consumes: Task 2's healthy stack.
- Produces: `paper/data/en_run_NN.json`, same schema as Task 2.

- [ ] **Step 1: Switch the pipeline to English**

Edit `.env`: `LANGUAGE=zh` → `LANGUAGE=en` (Edit tool — preserve comments/encoding). Then restart **the pipeline only**: read `local_services/config_panel/server.py` to find its pipeline-restart route and call it with `curl.exe -X POST http://localhost:7870/<route>`. Fallback if the panel isn't up: find the `:7860` PID via `netstat -ano | findstr ":7860.*LISTENING"`, terminate it (native `TerminateProcess` via the panel's own helper pattern — **not** `taskkill`, which hangs under load here), then start `python -m pipeline.main` with `run_in_background`. Wait until `http://localhost:7860/client/` returns 200 again.

- [ ] **Step 2: Verify the en question wav drives a turn**

```powershell
python -m scripts.measure --mic output/_en_q.wav
python -c "import json; d=json.load(open('output/measure_report.json',encoding='utf-8')); print(d['meta']['when'], repr(d['meta']['question']), d['meta']['ttfo'])"
```

Expected: `when` = now and `question` is English text. If the question is empty/zh/stale (the `q_ai.wav` failure mode), synthesize a known-good 16 kHz mono wav with Windows SAPI and re-verify with `--mic output/_en_q_new.wav`:

```powershell
Add-Type -AssemblyName System.Speech
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SetOutputToWaveFile("output/_en_q_new.wav", $fmt)
$s.Speak("What is artificial intelligence? Please explain it to me in detail.")
$s.Dispose()
```

- [ ] **Step 3: Run the campaign** — same loop as Task 2 Step 3 with the verified wav and target names `paper/data/en_run_$n.json`.

- [ ] **Step 4: Staleness check** — same as Task 2 Step 4 against `en_run_*.json`; also confirm each `meta.question` is English.

- [ ] **Step 5: Restore zh and verify**

Edit `.env` back to `LANGUAGE=zh`, restart the pipeline the same way, then run one throwaway `python -m scripts.measure --mic output/_zh_q_def.wav` and confirm `meta.question` is Chinese again. Append to `environment.md`: which wav the en runs used, and the restore confirmation.

- [ ] **Step 6: Commit**

```powershell
git add paper/data .env && git commit -m "paper: en TTFO campaign, 10 fresh-session runs (LANGUAGE restored to zh)"
```

(If `.env` ends up byte-identical after the restore, commit only `paper/data`.)

---

### Task 4: Aggregation script + VRAM table

**Files:**
- Create: `paper/scripts/aggregate.py`, `paper/data/summary.md` (generated), `paper/data/vram.md`

**Interfaces:**
- Consumes: `paper/data/{zh,en}_run_*.json` (Tasks 2–3 schema).
- Produces: `aggregate.py` runnable as `python paper/scripts/aggregate.py` (no args), writing `paper/data/summary.md`; `summary.md` sections `## zh (N runs)` / `## en (N runs)` each with a TTFO line (`median/p95/min/max`), a per-stage median table, per-run TTFO list, and an end-to-end-to-ear line. Tasks 5, 9, 11 read these numbers.

- [ ] **Step 1: Write `paper/scripts/aggregate.py`**

```python
"""Aggregate measure_report.json runs into paper/data/summary.md.

Run: python paper/scripts/aggregate.py
"""
import json
import statistics
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

# Waterfall rows as scripts.measure names them (cumulative to the ear).
STAGES = [
    "STT finalize -> LLM",
    "LLM -> TTS (sentence-1 flush)",
    "TTS synth first chunk",
    "TTS -> bot-start (steady lead-hold)",
    "Transport + encode + network",
    "Browser jitter + decode + playout",
]


def p95(xs):
    xs = sorted(xs)
    return xs[max(0, min(len(xs) - 1, round(0.95 * (len(xs) - 1))))]


def load(lang):
    return [(f.name, json.loads(f.read_text(encoding="utf-8")))
            for f in sorted(DATA.glob(f"{lang}_run_*.json"))]


def summarize(lang):
    runs = load(lang)
    ttfos = [(name, r["meta"]["ttfo"]) for name, r in runs if r["meta"].get("ttfo")]
    vals = [t for _, t in ttfos]
    lines = [f"## {lang} ({len(runs)} runs)", ""]
    lines.append(f"- TTFO median **{statistics.median(vals):.2f}s** / p95 {p95(vals):.2f}s"
                 f" / min {min(vals):.2f}s / max {max(vals):.2f}s")
    lines.append("- per-run: " + ", ".join(f"{n.split('_run_')[1][:2]}:{t:.2f}" for n, t in ttfos))
    ears = [r["raw"]["anchors"].get("playout") for _, r in runs
            if r.get("raw", {}).get("anchors", {}).get("playout")]
    if ears:
        lines.append(f"- end-to-end to the ear (probe arrival + jitter est): median "
                     f"{statistics.median(ears):.2f}s over {len(ears)} runs")
    lines += ["", "| stage | median delta (s) | n |", "|---|---|---|"]
    for st in STAGES:
        ds = [row["delta"] for _, r in runs for row in r.get("waterfall", [])
              if row["stage"] == st and row["delta"] is not None]
        cell = f"{statistics.median(ds):.2f}" if ds else "unknown"
        lines.append(f"| {st} | {cell} | {len(ds)} |")
    fps = [r["raw"]["probe"]["recv_fps"] for _, r in runs if r["raw"]["probe"].get("recv_fps")]
    frz = [r["raw"]["probe"]["freeze_ms"] for _, r in runs if r["raw"]["probe"].get("freeze_ms")]
    if fps and frz:
        lines += ["", f"- received video fps median {statistics.median(fps):.1f}; max freeze {max(frz)}ms", ""]
    else:
        lines += ["", "- probe video rows missing", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    out = ["# Measurement summary (generated by aggregate.py -- do not hand-edit)", ""]
    for lang in ("zh", "en"):
        if list(DATA.glob(f"{lang}_run_*.json")):
            out.append(summarize(lang))
    (DATA / "summary.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
```

- [ ] **Step 2: Run it and eyeball**

Run: `python paper/scripts/aggregate.py`
Expected: both `## zh` and `## en` sections, 20 per-run TTFOs, no `unknown` in the TTS/lead-hold rows. (`LLM first token` variance lives inside the sentence-1-flush row; the STT row is 0.0 by construction — the spec's waterfall.) If a stage errors on a missing key, fix `aggregate.py`, not the data.

- [ ] **Step 3: Capture the VRAM table**

With the full stack live, write `paper/data/vram.md` from:

```powershell
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
wsl -d Ubuntu -e bash -c "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || true"
```

Label rows by role (vLLM CosyVoice / MuseTalk avatar / pipeline / other). Expected ballpark from STATUS.md: whole card ~7.8 GB, TTS ~2.3 GB, avatar ~3.3 GB — flag in the file if reality is >1 GB off.

- [ ] **Step 4: Commit**

```powershell
git add paper/scripts/aggregate.py paper/data/summary.md paper/data/vram.md
git commit -m "paper: aggregate script, measurement summary, VRAM budget table"
```

---

### Task 5: Latency waterfall figure (Fig 2)

**Files:**
- Create: `paper/scripts/make_waterfall.py`, `paper/figures/waterfall.png`, `paper/figures/waterfall.pdf`

**Interfaces:**
- Consumes: `paper/data/{zh,en}_run_*.json` via the same STAGES list as `aggregate.py`.
- Produces: `paper/figures/waterfall.{png,pdf}` — one horizontal stacked bar per language, segments = median per-stage deltas, x-axis seconds, annotated with the TTFO median. Task 11 embeds `figures/waterfall.png`.

- [ ] **Step 1: Write `paper/scripts/make_waterfall.py`**

```python
"""Stacked per-stage latency waterfall (median deltas), zh vs en.

Run: python paper/scripts/make_waterfall.py
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
    ("LLM -> TTS (sentence-1 flush)", "LLM (to sentence-1 flush)"),
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
    ttfo = statistics.median([r["meta"]["ttfo"] for r in runs if r["meta"].get("ttfo")])
    return out, ttfo


def main():
    langs = [l for l in ("zh", "en") if list(DATA.glob(f"{l}_run_*.json"))]
    fig, ax = plt.subplots(figsize=(7.0, 1.2 + 0.9 * len(langs)))
    for yi, lang in enumerate(langs):
        deltas, ttfo = medians(lang)
        left = 0.0
        for (key, label), d, c in zip(STAGES, deltas, COLORS):
            ax.barh(yi, d, left=left, color=c, edgecolor="white",
                    label=label if yi == 0 else None)
            left += d
        ax.annotate(f"TTFO {ttfo:.2f}s", (left, yi), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9)
    ax.set_yticks(range(len(langs)), [l.upper() for l in langs])
    ax.set_xlabel("seconds after the user stops speaking (median of 10 fresh sessions)")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"waterfall.{ext}", dpi=300)
    print("wrote", FIGS / "waterfall.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate and inspect**

Run: `python paper/scripts/make_waterfall.py`, then Read `paper/figures/waterfall.png`.
Expected: two bars (ZH, EN), five segments each, TTFO annotations matching `summary.md` medians, legible at column width. Adjust figsize/fontsize if cramped.

- [ ] **Step 3: Commit**

```powershell
git add paper/scripts/make_waterfall.py paper/figures/waterfall.png paper/figures/waterfall.pdf
git commit -m "paper: latency waterfall figure (Fig 2)"
```

---

### Task 6: Architecture + sync diagrams (Fig 1, Fig 3)

**Files:**
- Create: `paper/figures/architecture.svg`, `paper/figures/proto2_sync.svg`

**Interfaces:**
- Produces: `figures/architecture.svg` (Fig 1, embedded by Task 8) and `figures/proto2_sync.svg` (Fig 3, embedded by Task 10).

- [ ] **Step 1: Author `architecture.svg`** (hand-written SVG, ~700px wide, white background, sans-serif ≥12px)

Content (from CLAUDE.md "Architecture — how one turn flows"): left-to-right boxes `Browser (WebRTC) → Silero VAD → STT (Deepgram / sherpa-onnx local) → LLM (OpenRouter / Ollama local) → CosyVoice TTS (vLLM, WSL, :8001) → MuseTalk avatar client ⇄ MuseTalk render server (TensorRT, :8002) → WebRTC out → Browser`. Annotate: dashed grouping box "single 16 GB consumer GPU" around TTS + render server; "streams: first sentence reaches TTS before the answer is complete" under the LLM→TTS edge; the ws wire contract on the client⇄server edge ("16 kHz PCM →, frames + video_start/clock/end ←, proto-2 header"); a "swap per stage via .env" note. Keep every label verifiable against CLAUDE.md — no invented components.

- [ ] **Step 2: Author `proto2_sync.svg`**

A two-lane timeline (audio lane, video lane) showing steady-mode release: TTS chunks arriving; frames arriving with the 16-byte header `MTF2 | kind u8 | audio_pos u64` (kind 0 real / 1 held / 2 idle); the client releasing buffered voice up to `audio_pos` of the last shown real frame; a "render stall → voice pauses (never drifts)" callout. Source: STATUS.md 25th-session §(1) and CLAUDE.md's proto-2 paragraph.

- [ ] **Step 3: Verify both render** — Read each SVG file (visual check: no overlapping text, arrows connect, correct spelling of MuseTalk/CosyVoice/Pipecat).

- [ ] **Step 4: Commit**

```powershell
git add paper/figures/architecture.svg paper/figures/proto2_sync.svg
git commit -m "paper: architecture and proto-2 sync diagrams (Fig 1, Fig 3)"
```

---

### Task 7: Related-work research → `references.bib` + §2 draft

**Files:**
- Modify: `paper/references.bib`, `paper/draft.md` (§2)

**Interfaces:**
- Produces: BibTeX keys the later sections cite. Fixed key list (later tasks use these exact keys): `musetalk2024, wav2lip2020, ditto2024, cosyvoice2_2024, cosyvoice3_2025, vllm2023, whisper2022, sileroVad, sherpaOnnx, pipecat, openavatarchat, livekitAgents, simli, heygen, opencc, tensorrt, aiortc, smartturn`.

- [ ] **Step 1: Web-search each reference** (WebSearch/WebFetch): find the real title/authors/year/venue or arXiv id for each key above. Verify titles against arXiv/GitHub — no guessed metadata. Software/products without papers get `@misc` with `howpublished = {\url{...}}` and a `note = {Accessed 2026-07}`.

- [ ] **Step 2: Fill `paper/references.bib`** with one verified entry per key. Run a duplicate/format sanity check: `python -c "content = open('paper/references.bib',encoding='utf-8').read(); import re; keys=re.findall(r'@\w+\{([^,]+),', content); assert len(keys)==len(set(keys)), 'dup keys'; print(len(keys), 'entries')"` — expected: 18 entries.

- [ ] **Step 3: Draft §2 Related Work (~350 words)** in `paper/draft.md`. Structure: (a) talking-head generation models (Wav2Lip, MuseTalk, Ditto) — offline/model-centric, not end-to-end conversational systems; (b) commercial real-time avatar APIs (Simli, HeyGen) — cloud-hosted, per-minute cost, data egress; (c) open conversational-avatar frameworks (OpenAvatarChat — closest neighbor; note its per-packet audio-in-frame coupling vs our metadata coupling, from STATUS.md 25th session §0) and orchestration layers (Pipecat, LiveKit Agents); (d) streaming TTS (CosyVoice 2/3 on vLLM). Position: none combine fully-local, single-consumer-GPU, multilingual, sub-~3s TTFO with structural A/V sync.

- [ ] **Step 4: Commit**

```powershell
git add paper/references.bib paper/draft.md
git commit -m "paper: verified references + Related Work section"
```

---

### Task 8: Draft §3 System Architecture

**Files:**
- Modify: `paper/draft.md` (§3; embed `figures/architecture.svg`)

**Interfaces:**
- Consumes: Fig 1; citation keys from Task 7.

- [ ] **Step 1: Write §3 (~850 words + Table 1)** covering, in order:
  1. **Pipeline shape** — the linear streaming pipeline (mic → VAD → STT → LLM → TTS → avatar → transport), built on Pipecat [@pipecat]; frames stream, so the LLM's first sentence reaches TTS before the full answer exists and TTS's first chunk reaches the avatar immediately. Fig 1 here.
  2. **Table 1: stage/provider matrix** — per stage: default provider, fully-local alternative, where it runs (from the CLAUDE.md stack table): VAD Silero (local); STT Deepgram nova-2 / sherpa-onnx zipformer (local CPU, zh→Traditional via OpenCC [@opencc]); LLM OpenRouter / Ollama; TTS CosyVoice 2/0.5B on vLLM [@vllm2023] in WSL2 / JaiTTS for Thai; avatar MuseTalk [@musetalk2024] + TensorRT [@tensorrt].
  3. **Design rule: thin single-provider stage factories** — each stage is one provider chosen by env config, deliberate fallback switches, not multi-provider branching; unknown value raises.
  4. **The avatar as a separate GPU process** — ws wire contract (16 kHz PCM in; RGB frames + video_start/clock/end out; proto-2 header — one sentence, detail deferred to §5); the turn taker vs renderer split; single shared 16 GB GPU with load-order and VRAM discipline (vLLM KV pool sized to the one-sentence-per-request workload at 0.07 util; torch weights freed after TensorRT engines load).
  5. **Client** — plain WebRTC browser page; optional split mode streaming only the mouth crop over the video track and compositing over a lossless background client-side.
- [ ] **Step 2: Claims check** — for each number/claim in §3, confirm it appears in CLAUDE.md/STATUS.md/`paper/data/` (grep the draft for digits; each must have a source). No unverifiable adjectives ("photoreal" is allowed as a description of MuseTalk's output class, cited).
- [ ] **Step 3: Commit** — `git add paper/draft.md && git commit -m "paper: System Architecture section"`

---

### Task 9: Draft §4 Latency Engineering

**Files:**
- Modify: `paper/draft.md` (§4; embed `figures/waterfall.png`; Table 2 ablations)

**Interfaces:**
- Consumes: `paper/data/summary.md` medians (Task 4), Fig 2 (Task 5).

- [ ] **Step 1: Write §4 (~850 words + Fig 2 + Table 2)** covering:
  1. **The metric** — TTFO = user-stops-speaking → first synced audio+video reaching the client; plus the to-the-ear waterfall methodology: same-box clock stitch (log `t0.timestamp()` == probe `time.time()`), per-stage rows summing to a true end-to-end (from the measure design spec `docs/superpowers/specs/2026-07-06-measure-end-to-end-latency-design.md`).
  2. **Fig 2** with the fresh medians; walk one language through the stages.
  3. **The levers, each one paragraph with its measured before/after** (Table 2 summarizes): (a) TTS autoregressive LM moved to vLLM (first-chunk ~3.4→~1.1 s — the single largest win); (b) first-clause splitting, per-language: en char-window comma/space split 18/32 (TTS first chunk ~3.0→~1.7 s, TTFO ~4.6→~3.2 s), zh full-width-comma-only split, never a char cap (long-opener turns 4.78→3.08 s) — and why the language difference exists (prefill scales with input sentence length; zh lacks ASCII commas/spaces); (c) LLM provider pinning (zh LLM hop 1.64→0.80 s median, tail eliminated); (d) the lead-frame cushion as the deliberate sync-vs-start tradeoff (14 frames; every lower value rejected by live perception — foreshadows §7).
  4. **What the metric cannot see** — VAD stop-secs and turn-end detection precede t0 (perceived but unmeasured), honest per the spec.
- [ ] **Step 2: Claims check** — every number in §4 must match `summary.md` (fresh) or the P-record (historical, cited as "measured during development"). The two classes must be visually distinct in the text (fresh numbers reference the evaluation setup; historical A/Bs say so).
- [ ] **Step 3: Commit** — `git add paper/draft.md && git commit -m "paper: Latency Engineering section"`

---

### Task 10: Draft §5 Audio–Visual Synchronization

**Files:**
- Modify: `paper/draft.md` (§5; embed `figures/proto2_sync.svg`)

**Interfaces:**
- Consumes: Fig 3 (Task 6); STATUS.md 25th-session P51 material.

- [ ] **Step 1: Write §5 (~800 words + Fig 3)** covering:
  1. **The problem** — a render process on a contended GPU cannot guarantee fps; audio-master ("live") lets lips trail; naive video-master pairing by frame index breaks the moment server and client disagree about fps or a frame is a held re-send.
  2. **steady (video-master) sync** — voice buffered and released paced to the frames the server actually rendered; a render stall pauses the voice rather than drifting; synced start via the lead-frame cushion.
  3. **proto 2: the frame declares itself** — the 16-byte header (`MTF2 | kind | audio_pos`), audio_pos = cumulative real 16 kHz samples covered once that frame shows; the client releases voice paired to the renderer's own account, so an fps mismatch structurally cannot shift the mapping and held/idle frames are declared, not guessed. Contrast with OpenAvatarChat's audio-bytes-in-packet coupling [@openavatarchat]: metadata was chosen so the delivered voice stays the original 24 kHz TTS audio rather than the 16 kHz lip-sync copy.
  4. **Keeping the renderer ≥ fps** — TensorRT UNet+VAE path (per-segment render 389→255 ms) as the enabler that makes video-master viable under shared-GPU contention; cudnn.benchmark pitfall one sentence.
  5. **Verification** — probe invariant (final audio_pos == fed samples exact, 90970/90970), offline capture end-drift ±0.04 s, and the human live-eye gate as final arbiter (foreshadows §7).
- [ ] **Step 2: Claims check** — same grep-the-digits pass; proto-2 details must match STATUS.md §(1)/(2) exactly (16-byte header, kind values, opt-in negotiation).
- [ ] **Step 3: Commit** — `git add paper/draft.md && git commit -m "paper: A/V Synchronization section"`

---

### Task 11: Draft §6 Evaluation

**Files:**
- Modify: `paper/draft.md` (§6; Tables 3–4)

**Interfaces:**
- Consumes: `paper/data/summary.md`, `paper/data/vram.md`, `paper/data/environment.md`.

- [ ] **Step 1: Write §6 (~700 words + Table 3 TTFO/waterfall results + Table 4 VRAM budget)**:
  1. **Setup** — hardware (RTX 5060 Ti 16 GB, Windows 11 + WSL2), the exact config from `environment.md`, the measurement protocol (10 fresh sessions per language, headless WebRTC probe, why fresh sessions: avoids the known session-degradation confound — stated honestly), what median vs p95 captures (p95 includes provider congestion).
  2. **Table 3** — TTFO median/p95 per language + the per-stage median waterfall + end-to-end-to-ear estimate, all from `summary.md` verbatim.
  3. **Sync results** — received-fps and freeze numbers from `summary.md`; end drift ±0.04 s from the P51 baseline (labeled as the development-time measurement it is).
  4. **Table 4** — VRAM per process from `vram.md`; the ~7 GB project share on a 16 GB card; one sentence on what that implies (dedicated-GPU headroom / 8 GB-card feasibility with the documented settings).
  5. **Ablation recap table or inline** — pointer back to Table 2 rather than duplicating it (DRY).
- [ ] **Step 2: Cross-check every cell** of Tables 3–4 against the generated files — no transcription by memory; copy values from `summary.md`/`vram.md` directly.
- [ ] **Step 3: Commit** — `git add paper/draft.md && git commit -m "paper: Evaluation section"`

---

### Task 12: Draft Abstract, §1 Introduction, §7 Discussion & Lessons, §8 Conclusion

**Files:**
- Modify: `paper/draft.md`

**Interfaces:**
- Consumes: all drafted sections; the spec's contributions list and lessons list.

- [ ] **Step 1: Write §7 Discussion and Lessons Learned (~400 words)** — the three lessons from the spec, each grounded in its concrete episode: (a) *the probe passes what the eye rejects — in both directions* (lead-frame sweep rejected by live perception despite clean probes; conversely a measured zh waveform delta that never reached the eye — instrument deltas are neither necessary nor sufficient); (b) *the reference must not share the suspect input* (the deterministic-render check that could not fail); (c) *restore invariants at the producer* (PCM sample alignment enforced once at chunk creation instead of patched at every consumer). Then **Limitations** honestly: single-client server; mouth region model-bound to 256 px; perceptual judgments by a single expert viewer, no formal MOS study; Thai supported via a separate TTS, not the primary voice; remaining session-length degradation under investigation (drop this clause if resolved by drafting time — check STATUS.md).
- [ ] **Step 2: Write §1 Introduction (~500 words)** — motivation (embodied conversational agents; cloud avatar cost/privacy/latency; consumer-hardware accessibility), the one-sentence claim from the spec verbatim in spirit, the four numbered contributions from the spec, paper roadmap.
- [ ] **Step 3: Write the Abstract (~180 words) and §8 Conclusion (~150 words)** — abstract: problem, system, the headline numbers (fully local, ~7 GB on one consumer GPU, TTFO medians from Table 3, multilingual), the proto-2 sync idea, availability of the code (public repo — phrase as "code available" only if the user confirms the repo link goes in).
- [ ] **Step 4: Commit** — `git add paper/draft.md && git commit -m "paper: abstract, intro, discussion, conclusion"`

---

### Task 13: Full-paper review + preview for the user

**Files:**
- Modify: `paper/draft.md` (fixes only)
- Create: scratchpad HTML preview (not committed)

**Interfaces:**
- Consumes: the complete draft.

- [ ] **Step 1: Consistency pass** — one read of the full draft checking: (a) every `[@key]` exists in `references.bib` (script it: extract `\[@(\w+)\]` from draft, compare to bib keys, print unmatched — expect none); (b) every number appears in `paper/data/` or names its P-record provenance; (c) section budgets roughly hold (total prose ≤ ~4,500 words: `python -c "import re; t=open('paper/draft.md',encoding='utf-8').read(); print(len(re.sub(r'<!--.*?-->','',t,flags=re.S).split()))"`); (d) terminology uniform (TTFO, steady/live, proto 2 — pick one spelling each); (e) no leftover `<!-- Task N -->` markers.
- [ ] **Step 2: Fix anything found, commit** — `git commit -am "paper: full-draft consistency pass"`.
- [ ] **Step 3: Render a preview** — build a single-file HTML rendering of `draft.md` (figures inlined) in the scratchpad and publish it as a private Artifact so the user reads the paper as a formatted page, not raw Markdown.
- [ ] **Step 4: Hand off** — final message: where the draft lives, the headline fresh numbers, the open items that are the user's (author names/affiliations, push/publish decision, venue question for the prof, whether "code available" goes in), and that LaTeX IEEEtran conversion is the next phase once the prof approves content.
