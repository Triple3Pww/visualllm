# NCU Handoff Repo — Design Spec

_Date: 2026-07-18 · Status: approved, ready for planning_

## Goal

Produce **`VisualLLm-NCU`**, a clean, curated, self-contained copy of the working
real-time avatar system that NCU (National Central University, Taiwan) researchers and
students can **set up, run, and extend** without the 30-session archaeology of the
development repo.

The current repo (`E:\Claude\VisualLLm`) is a working system buried under accumulated
session logs (a 1,600-line `STATUS.md`, a large `PROBLEMS-AND-FIXES.md`, and folders of
experiments). The handoff repo keeps the working system, drops the out-of-scope
subsystems, **cleans the code that carries over**, and replaces the session-log docs with
a small set of docs written for an outside reader.

## Non-goals

- Not a rewrite. Working logic is preserved; cleanup is surgical (remove orphans + dead
  code, not gratuitous refactoring).
- Not a re-architecture. The pipeline, the avatar wire contract, and the A/V-sync design
  are carried over as-is.
- Not a fork with shared history. Fresh `git init`, single initial commit.

## Audience & what they need

NCU wants to **use** the system and **future-implement** on it. Their specific interest
(from the existing collaboration) is the Chinese/English path and plugging their own LLM
backend (the `resWeatherChain` LangServe endpoint) into the LLM slot. So the handoff must
make the **stage-swap pattern** and the **"connect your own LLM"** path first-class and
well-documented, using `weather_chain` as the worked example.

## Scope

### Included (the working system, cleaned)

| Area | Carries over |
|------|--------------|
| Pipeline | `pipeline/` — VAD (Silero) → STT → LLM → TTS → avatar → `TtfoMeter`, the thin per-stage factories, `config.py`, `main.py`, `metrics.py` |
| STT | sherpa-onnx local offline **+** Deepgram cloud, `.env`-switchable |
| LLM | OpenRouter (cloud / local Ollama) **+ `weather_chain` kept as the worked example** of wiring a custom non-OpenAI backend |
| TTS | CosyVoice on vLLM-in-WSL — `tts/cosyvoice-server/` wrapper code (upstream checkout + weights gitignored) |
| Avatar | MuseTalk server + `musetalk_video.py`; proto-2 wire; steady/live sync; **mouth-crop split (`MUSETALK_SPLIT`) kept**; **avatar preset system kept** |
| Clients | `/studio` custom client + untouched `/client` prebuilt fallback |
| Config panel | `local_services/config_panel/` (`:7870`) |
| Measure harness | `scripts/measure/` — **kept** |
| Launch tooling | `scripts/launch.ps1`, `run.ps1`, the `.exe` shim, `preflight.py`, probes |

### Removed

- Growing-memory harness: `local_services/avatar_memory.py`, `state/avatar_memory/`, and
  its wiring in `main.py` / `config.py`.
- Thai JaiTTS: `local_services/jaitts_server/`, the `jaitts` branch in `stages/tts.py`.
- `local_services/nimbus_client/` and its installer (superseded by `/studio`).
- Dev-only trees: `archive/`, `learn/`, `research/`, `scratchpad_tts/`, `output/`,
  `paper/`, `tools/`, `state/`.
- Session-log docs: `STATUS.md`, `PROBLEMS-AND-FIXES.md`, `PROBLEMS-AND-FIXES-CLEAN.md`,
  all `*-HANDOFF.md`, `WORKFLOW.md` (folded into the new docs), the old `README`/`SETUP`.

Note: `sherpa_stt.py` stays (STT, unrelated to the removed Thai TTS).

## Code-cleanup pass (per carried-over module)

Every module that carries over is read and cleaned:

1. **Remove references to dropped subsystems** — `avatar_memory` wiring, the `jaitts` TTS
   branch, the `nimbus_client` installer; confirm the already-removed
   `moss`/`elevenlabs`/`funasr` leave no stragglers.
2. **Remove phantom/dead knobs** — `.env` keys with no reader; dead one-off diagnostics.
   Keep `MUSETALK_DUMP_*` (the only trustworthy mouth-debug path per the P40 lesson).
3. **Tidy stale comments** — fix/drop comments pointing at removed code; **preserve the
   load-bearing "why" comments** (latency, Pipecat quirks, hardware constraints).
4. **Clean `.env.example`** to the reduced scope.

Known reference sites (from grep, to be verified during execution):
`local_services/config_panel/server.py`, `pipeline/stages/stt.py`, `pipeline/config.py`,
`pipeline/main.py`, `pipeline/stages/tts.py`.

## Repository structure

```
VisualLLm-NCU/
├─ README.md                 # what it is, architecture-at-a-glance, repo map, quickstart pointer
├─ pipeline/                 # Pipecat pipeline: stages/, config.py, main.py, metrics.py
├─ local_services/           # avatar server + video service, STT, TTS client, config panel, studio client
├─ tts/cosyvoice-server/     # CosyVoice vLLM wrapper (upstream checkout + weights gitignored)
├─ scripts/                  # launch/run, measure/, preflight, probes
├─ assets/                   # avatar portraits + voice refs
├─ models/                   # (gitignored) sherpa STT model — documented download
├─ docs/
│  ├─ ARCHITECTURE.md
│  ├─ SETUP.md
│  ├─ USAGE.md
│  ├─ EXTENDING.md
│  └─ ENGINEERING-NOTES.md
├─ .env.example              # cleaned to reduced scope
├─ requirements.txt
└─ .gitignore
```

## Documentation set (written for an outside reader)

- **README.md** — one-paragraph what-it-is, the stage table, an architecture-at-a-glance
  diagram, a repo map, and a pointer to SETUP. The front door.
- **docs/ARCHITECTURE.md** — how one turn flows; the thin stage-factory pattern; the
  avatar as a separate GPU process; the client↔server wire contract (proto 2); A/V sync
  (`steady` vs `live`); the streaming/TTFO design.
- **docs/SETUP.md** — from zero: hardware requirements, WSL2 + vLLM CosyVoice, the conda
  envs (pipeline / musetalk / tts), weights + models download, `.env`, run, verify
  (`preflight`), troubleshooting.
- **docs/USAGE.md** — the three processes and the launcher; the config panel; `/studio`
  vs `/client`; the important `.env` knob reference (curated subset, not all 100+).
- **docs/EXTENDING.md** — the "future implement" doc: how to swap/add a stage using the
  factory pattern; **the `weather_chain` worked example** of connecting a custom LLM
  backend; how to add an avatar preset.
- **docs/ENGINEERING-NOTES.md** — distilled, still-relevant hard-won lessons, grouped:
  *Audio integrity* (odd-byte PCM fixed at the producer; judge garble on a concatenated
  WAV), *Avatar/GPU* (`cudnn.benchmark=False`; TRT buys margin; VRAM budget; load order),
  *Networking* (WSL IP not localhost; never block the event loop; TURN/ICE), *Latency
  methodology* ("the probe passes what the eye rejects"; measure a fresh session; the
  pre-t0 blind spot). Raw `PROBLEMS-AND-FIXES.md` is dropped.

## Heavy-asset policy

Gitignored + documented-download, never bundled:
- upstream `CosyVoice/` (~17 GB) + its weights
- MuseTalk vendor weights and its nested `.git`
- sherpa-onnx STT model (`models/`)
- TensorRT engine cache (`musetalk_server/trt_cache/`)

Repo carries source only; `docs/SETUP.md` holds every fetch command.

## Verification (before hand-back)

1. `python -m scripts.preflight` passes in `VisualLLm-NCU` (all fragile Pipecat imports
   resolve).
2. All carried-over modules import without referencing removed subsystems (grep clean).
3. Drive at least one real turn through the stack from the new repo (measure harness) and
   report the result. If a full live-eye run is not possible from here, say so plainly —
   cleaning a working system is verified by behavior, not by looking clean.

## Risks

- **Cleanup breaks a working path.** Mitigation: surgical edits + the verification gate;
  keep the source repo untouched as the reference.
- **A "dead" knob is actually load-bearing.** Mitigation: grep for a reader before
  removing; the P40/`MUSETALK_DUMP_*` and `cudnn.benchmark` lessons are explicit keeps.
- **Path assumptions** (WSL `/mnt/e/Claude/VisualLLm/...`, absolute asset paths) baked in
  code/scripts. Mitigation: audit and re-point to the new repo root during cleanup.

## Open items (defaults chosen, correctable)

- Repo name `VisualLLm-NCU` at `E:\Claude\VisualLLm-NCU`.
- Removed list as above.
