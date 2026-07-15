# Design: IEEE paper on the VisualLLm architecture

_Date: 2026-07-15 · Branch: chore/cleanup-and-tts-merge_

## Goal

Produce an IEEE-format paper on the VisualLLm system architecture, requested by the professor
after approving the system's live performance. Decisions already made with the user:

- **Venue-agnostic**: the prof said only "IEEE paper" — draft in standard IEEE **conference**
  format (two-column, ~6 pages). It upgrades cleanly to a journal draft or downgrades to a
  report once the prof names a target.
- **Contribution = the whole-system architecture** (a systems paper), not a single-technique
  paper. Latency and A/V sync become sections, not the spine.
- **Markdown first**: draft and iterate in Markdown; convert to LaTeX IEEEtran only after the
  prof approves the content. Collect references as BibTeX from day one so the conversion is
  mechanical.
- **No hard deadline; fresh measurement runs are allowed** (the system is up and the measure
  harness works).

## The one-sentence claim

> A complete speech → STT → LLM → TTS → photoreal talking-head agent can run **fully streaming,
> multilingual (en/zh, th via a swappable TTS), and fully local** on a **single consumer 16 GB
> GPU (~7 GB used)** with time-to-first-output **~3 s**, achieved by pipeline/system engineering
> on top of existing open models rather than by training new ones.

Everything in the paper serves that claim. The differentiators vs published/commercial systems:
fully local (no per-minute cloud avatar bill, no data egress), consumer hardware, multilingual
with per-language latency levers, and a sync design where drift is structurally impossible.

## Paper skeleton (6-page IEEE conference budget)

| § | Section | Budget | Content |
|---|---------|--------|---------|
| 1 | Introduction | ~0.75 p | Motivation (real-time embodied agents; cloud cost/privacy/latency), the claim, numbered contributions |
| 2 | Related Work | ~0.5 p | Cloud avatar APIs (Simli, HeyGen-class); open frameworks (OpenAvatarChat, LiveTalking); talking-head models (MuseTalk, Wav2Lip, Ditto); streaming TTS (CosyVoice 2/3); orchestration (Pipecat, LiveKit) |
| 3 | System Architecture | ~1.25 p | Pipeline diagram; streaming frame flow (first LLM sentence reaches TTS before the answer exists); thin single-provider stage factories; the avatar as a separate GPU process + ws wire contract; local/cloud swap per stage |
| 4 | Latency Engineering | ~1 p | TTFO definition + the to-the-ear waterfall methodology; the levers with measured before/after: vLLM-hosted TTS AR model (3.4→1.1 s TTFB), per-language first-clause splitting (en char-bounded, zh comma-boundary-only), LLM provider pinning (1.64→0.80 s zh), lead-frame cushion |
| 5 | A/V Synchronization | ~1 p | steady (video-master) vs live (audio-master); proto-2 per-frame header (kind + cumulative audio_pos) — the server's own account of rendered audio replaces index arithmetic, fps mismatch cannot shift the mapping; TensorRT render path keeping ≥fps under shared-GPU contention |
| 6 | Evaluation | ~1 p | Setup (RTX 5060 Ti 16 GB, Windows+WSL2); fresh TTFO runs per language; per-stage waterfall; sync drift; VRAM budget table; ablation table from recorded A/Bs |
| 7 | Discussion & Lessons | ~0.5 p | The 2–3 strongest engineering lessons (below) + honest limitations |
| 8 | Conclusion + refs | ~0.25 p | |

**Contributions list (§1), numbered:**
1. An end-to-end streaming architecture for a fully-local multilingual talking-head agent on one
   consumer GPU, with per-stage local/cloud substitution.
2. Per-language time-to-first-output engineering and a per-stage latency-to-the-ear measurement
   methodology (the waterfall that sums to a true end-to-end number).
3. A drift-free A/V sync mechanism: video-master pacing driven by per-frame audio-position
   metadata declared by the renderer (proto 2).
4. Resource engineering that fits TTS-LLM (vLLM) + avatar rendering (TensorRT) in ~7 GB shared
   VRAM.

**Lessons for §7 (from the P-record, folded in per the approved approach):**
- *The probe passes what the eye rejects — in both directions* (P19/P33): instrument-level
  deltas are neither necessary nor sufficient for perceptual defects; the live eye is the
  arbiter.
- *Metrology: the reference must not share the suspect input* (P40): a deterministic-render
  check fed the same corrupt PCM on both sides cannot fail.
- *Restore invariants at the producer, not the consumers* (P52): sample alignment enforced once
  where bytes are created, instead of patched at every consumer.

**Limitations (stated honestly):** single-client server; the animated mouth is model-bound to
256 px; no formal user study (single-expert perceptual judgment); the session-degradation issue
if still unresolved at submission time; zh leading-breath tradeoffs.

## Evaluation plan (the fresh runs)

All via the existing harness — no new measurement code:

- **TTFO**: `python -m scripts.measure --offline-capture`, **`--mic output/_zh_q_def.wav`** for
  zh (`q_ai.wav` no longer transcribes on Deepgram — the 27th-session landmine); for en, switch
  `.env` `LANGUAGE=en` and use an English question wav (synthesize one via CosyVoice if none in
  `output/` transcribes cleanly). **N=10 per language, fresh sessions** (the session-degradation bug means a long
  session mis-attributes; the measure doc says distrust single runs — Groq congestion). Report
  median + p95.
- **Waterfall**: per-stage rows from `output/measure_report.json` (STT-fold, LLM, TTS
  first-chunk, steady-hold, last mile).
- **Sync**: lips-start offset + end drift from the measure avatar rows (the P51 baseline
  measured ±0.04 s end drift).
- **VRAM**: `nvidia-smi` per-process table (vLLM TTS ~2.3 GB, avatar ~3.3 GB, whole card
  ~7.8 GB live).
- **Ablations — from the recorded A/Bs, cited as prior measurements, not re-run**: first-piece
  split (zh long-opener 4.78→3.08 s), Groq pin, vLLM vs Windows-PyTorch TTS, TRT vs PyTorch
  render (389→255 ms/segment), lead<14 rejection. Re-run only if a number looks stale during
  drafting.

**Figures/tables:** Fig 1 architecture diagram; Fig 2 latency waterfall (stacked bars per
language); Fig 3 proto-2 sync timeline (header fields + release pairing); optional Fig 4
`/studio` screenshot. Tables: stage/provider matrix; TTFO results; ablations; VRAM budget.

## Repository layout & workflow

- `paper/draft.md` — the single Markdown draft (sections in order; keeps diffing simple).
- `paper/references.bib` — BibTeX collected from the start (web research for related work
  happens during implementation, via the web-search agent).
- `paper/figures/` — generated figures (the waterfall can be produced from
  `measure_report.json`; the architecture diagram drawn as SVG/mermaid → PNG).
- LaTeX conversion (IEEEtran) is a **separate later phase**, after the prof approves content —
  not in this plan's scope beyond keeping the draft conversion-friendly (BibTeX keys, figure
  files, no Markdown-only constructs in critical places).

## Constraints & risks

- **Public repo**: `main` is public. The paper draft in `paper/` will become public if merged
  to main — ask the prof/user before pushing if pre-publication secrecy matters; the memory
  benchmark branch stays out of the paper entirely (separate work, per the standing rule).
- **Numbers drift**: the branch has uncommitted work; freeze a measurement baseline (commit or
  tag the state the numbers were taken on) so the paper's numbers are reproducible.
- **Related-work coverage** is the one area with no existing material in-repo — it needs real
  literature search (MuseTalk/CosyVoice/Pipecat papers exist; commercial systems need careful
  citation).
- **Authorship/affiliation** details (prof's name, NCU/Mahidol affiliations, author order) are
  the user's to supply — placeholder in the draft until then.

## Approaches considered

- **A (chosen): classic systems paper + fresh evaluation campaign**, with the best engineering
  lessons folded into §7.
- B: lessons-first experience paper — rejected as the spine (venue risk), kept as §7 content.
- C: descriptive report from existing numbers only — rejected; wastes the working harness and
  needs rework when a venue appears.
