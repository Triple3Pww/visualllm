# VisualLLm — Speech → LLM → Talking-Head Avatar

A real-time conversational system: **speak to it, and a photoreal 2D avatar
speaks back** (lip-synced audio + video). Multi-turn, with barge-in.

```
speech → STT → LLM → TTS → lip-sync avatar → audio+video out
```

**Goal:** time-to-first-output (you stop speaking → avatar starts responding)
**< 8 seconds**. See `docs`/the design plan for the full latency budget.

The whole system streams: as soon as the LLM emits its first sentence it flows
to TTS → first audio chunk → the avatar starts talking. We never wait for a
stage to fully finish.

---

## Architecture

Built on **[Pipecat](https://github.com/pipecat-ai/pipecat)** — it wires every
stage with streaming + barge-in built in, and ships pluggable services for all
the providers below. Every stage is selected from `.env` (no code changes to
swap English↔Mandarin or API↔local).

| Stage | Default (EN prototype) | Mandarin / local target |
|-------|------------------------|-------------------------|
| VAD / turn-taking | Silero (local) | same |
| STT   | Deepgram | FunASR Paraformer / faster-whisper |
| LLM   | GPT-4o-mini | Qwen2.5-7B (local via vLLM/Ollama) |
| TTS   | ElevenLabs Flash | CosyVoice2-0.5B (local) |
| Avatar| Simli / HeyGen LiveAvatar | MuseTalk (local, 5060 Ti) |
| Transport | WebRTC → browser | same |

```
pipeline/
  main.py            pipeline assembly + dev runner
  config.py          provider selection + env (single source of truth)
  metrics.py         TtfoMeter — measures time-to-first-output
  stages/            one factory per stage (isolated, swappable imports)
local_services/      local model wrappers (Phase 3)
client/              browser WebRTC client
scripts/
  bench_latency.py   per-stage latency probes
```

---

## Quick start (Phase 1 — English prototype)

> **Which APIs to buy + where each key/ID comes from:** see
> [`docs/SETUP.md`](docs/SETUP.md) — dedicated provider per stage (4 services,
> all with free tiers).

```bash
python -m venv .venv
.venv\Scripts\activate            # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m scripts.preflight       # verify imports resolve (catches Pipecat
                                  # version drift) BEFORE wiring keys

copy .env.example .env            # then fill in keys (Deepgram, OpenAI,
                                  # ElevenLabs, Simli)
python -m pipeline.main
```

Open the printed `http://localhost:7860` URL, allow the mic, and talk. The
console logs a `[TTFO]` line per turn; the disconnect log prints the
median/p95 summary.

> **Version note:** Pipecat's import paths shift between releases. If an import
> errors, check `python -c "import pipecat; print(pipecat.__version__)"` and
> adjust — the fragile imports are isolated to `pipeline/stages/*.py`,
> `pipeline/main.py`, and `pipeline/metrics.py`.

---

## Switching to Mandarin (Phase 2)

Edit `.env`:

```
LANGUAGE=zh
STT_PROVIDER=whisper_local     # or funasr
LLM_PROVIDER=qwen_local        # vLLM/Ollama OpenAI-compatible endpoint
TTS_PROVIDER=cosyvoice_local
```

(Avatar stays on the API — Simli/HeyGen both lip-sync zh audio fine.)

## Going local (Phase 3)

Stand up the local model servers in `local_services/` on the 5060 Ti and point
`COSYVOICE_URL` / `MUSETALK_URL` / `QWEN_BASE_URL` at them. Watch VRAM —
16 GB does **not** fit MuseTalk + CosyVoice2 + FunASR + Qwen-7B all at once, so
keep either the LLM or the avatar on API in steady state (tune in Phase 3).

---

## Measuring the goal

- **End-to-end:** `TtfoMeter` (in the pipeline) logs each turn's TTFO and a
  p95 summary. **Pass = p95 < 8 s.**
- **Per-stage:** `python -m scripts.bench_latency --stage llm` /`--stage tts`
  to find the dominant cost.
- Biggest tuning lever: the VAD `stop_secs` in `pipeline/stages/vad.py`.
