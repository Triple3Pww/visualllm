# VisualLLm — Speech → LLM → Talking-Head Avatar

A real-time conversational system: **speak to it, and a photoreal avatar speaks
back** (lip-synced audio + video). Multi-turn, streaming end-to-end.

```
speech → STT → LLM → TTS → lip-sync avatar → audio+video out
```

**Goal:** time-to-first-output (you stop speaking → avatar starts responding)
**< 8 seconds**.

The whole system streams: as soon as the LLM emits its first sentence it flows
to TTS → first audio chunk → the avatar starts talking. We never wait for a
stage to fully finish.

---

## Architecture

Built on **[Pipecat](https://github.com/pipecat-ai/pipecat)** — it wires every
stage with streaming + barge-in built in. This is one pure stack (no
provider-switching sprawl); the `.env` knobs are `LANGUAGE` (en/zh/th),
`TTFO_TARGET_SECONDS`, `AVATAR` (`ditto`|`none`), and `CHARACTER_MODE`.

| Stage | Service |
|-------|---------|
| VAD / turn-taking | Silero (local) |
| STT   | Deepgram (nova-2; `en-US` / `zh-TW` / `th` by `LANGUAGE`) |
| LLM   | OpenRouter (any model via `OPENROUTER_MODEL`) |
| TTS   | ElevenLabs (flash_v2_5, multilingual) |
| Avatar| Ditto — local lip-sync server on the GPU (5060 Ti); audio frame-clocked to the rendered video. Or `AVATAR=none` (client renders the face) |
| Transport | WebRTC → browser |

```
pipeline/
  main.py            pipeline assembly + dev runner
  config.py          keys + the en/zh/th switch (single source of truth)
  metrics.py         TtfoMeter — measures time-to-first-output
  stages/            one factory per stage (stt/llm/tts/avatar/vad)
local_services/      local avatar server + Pipecat wrapper
scripts/
  preflight.py       resolve every import (catches Pipecat version drift)
```

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m scripts.preflight       # verify imports resolve BEFORE wiring keys

copy .env.example .env            # then fill in keys (Deepgram, OpenRouter,
                                  # ElevenLabs)
```

The avatar runs as a **separate local server** (its own `ditto` conda env).
Start it first, then the pipeline:

```bash
conda run -n ditto python -m local_services.ditto_server.app   # GPU avatar server, :8002
python -m pipeline.main                                        # serves /client
```

Open the printed `http://localhost:7860/client` URL, allow the mic, **wait for
the avatar face to appear**, then talk. The console logs a `[TTFO]` line per
turn; the disconnect log prints the median/p95 summary.

> **Version note:** Pipecat's import paths shift between releases. If an import
> errors, check `python -c "import pipecat; print(pipecat.__version__)"` — the
> fragile imports are isolated to `pipeline/stages/*.py`, `pipeline/main.py`,
> and `pipeline/metrics.py`.

## Switching to Mandarin

Set `LANGUAGE=zh` in `.env` (and optionally an `OPENROUTER_MODEL` strong at
Chinese). Deepgram switches to `zh-TW` and ElevenLabs flash_v2_5 speaks zh — no
code changes.

---

## Measuring the goal

- `TtfoMeter` (in the pipeline) logs each turn's TTFO and a p95 summary.
  **Pass = p95 < 8 s.**
- Biggest tuning lever: the VAD `stop_secs` in `pipeline/stages/vad.py`.
