# VisualLLm — Project Status & Next Steps

_Last updated: 2026-06-10_

## 🆕 Phase 3 done: local MuseTalk avatar (replaces Simli)
The avatar now runs **locally on the 5060 Ti** — no more Thailand→US cloud lag.
`AVATAR_PROVIDER=musetalk_local` in `.env`. It's **two processes**: start the
MuseTalk server (`local_services\musetalk_server\run_server.bat`, in the dedicated
`musetalk` conda env) **first**, then `python -m pipeline.main`. Verified headless
(engine renders ~1.14× realtime at 20 fps; websocket returns correct 512² frames;
`/health` ok; pipeline serves `/client`). **Still needs your in-browser check of
lip-sync + audio/video sync**, and swap `assets/avatar.png` for your own face
(then delete `local_services/musetalk_server/avatar_cache/`). Build details +
the audio/video-sync caveat are in `PLAN.md`. Everything below describes Phase 1.

## What this is
Real-time **speech → LLM → photoreal talking-head avatar** system. Multi-turn,
barge-in, streaming end-to-end. Goal: time-to-first-output **< 8 s**. Built on
**Pipecat 1.3.0**, WebRTC to the browser. English prototype now; **Mandarin
(zh-TW)** is the real research target.

Currently running from a **remote Windows PC in Thailand**, accessed via **RDP**
(laptop mic redirected in as the "Remote Audio" input device).

## ✅ Working now (Phase 1 done)
- Full pipeline live: **Deepgram STT → OpenRouter (Gemini 2.5 Flash Lite) →
  ElevenLabs TTS → Simli avatar**, WebRTC → browser at `/client`.
- Avatar shows + lip-syncs; voice plays. Measured **TTFO median 1.97 s, p95
  2.86 s** — well under 8 s.
- LLM connection is **pre-warmed** on connect (cuts first-turn cold start).
- Custom **loading overlay** on the prebuilt UI: shows "Loading the avatar…"
  until the avatar's video actually appears (fixes the premature
  "Waiting for messages…").

## How to run
```
cd E:\Claude\VisualLLm
python -m pipeline.main            # serves http://localhost:7860/client
```
Open `/client` in the RDP desktop browser, **hard-refresh (Ctrl+Shift+R)**, click
through, **wait for the avatar face to appear**, then talk. Keys are in `.env`
(gitignored). Verify imports anytime with `python -m scripts.preflight`.

## ⚠️ Known issues / decisions pending
1. **Simli (US cloud) ~10–15 s warmup per connection**, occasional video stutter
   (`Vp8Decoder failed`) and rare total session failure ("participant
   undefined") — all caused by the **transpacific link from Thailand to Simli's
   US servers**. The loading overlay *hides* the wait but can't remove it.
   - The `MicGate` experiment (pipeline/gate.py) was **removed** — it was
     unreliable (fell back to its 30 s safety timer). File still exists, unused.
2. **Free-tier limits**: Simli ~50 min/month, ElevenLabs voice list was empty
   (using premade voice **Adam** `pNInz6obpgDQGcFmaJgB`, the one that worked).
3. **OpenRouter key was briefly committed in `.env.example`** → should be
   **rotated** at openrouter.ai.

## ▶️ Next steps (in priority order)
1. **Verify the loading overlay** works after hard-refresh (was the last change;
   untested by user). If the overlay never clears, check `pipeline/main.py`
   bottom block (the JS polls for a playing `<video>`).
2. **THE REAL FIX — local MuseTalk avatar (Phase 3).** Removes the cloud warmup,
   stutter, failures, and quota permanently — biggest win for Thailand latency.
   Work required:
   - Download MuseTalk weights (~5 GB) + a portrait into `assets/avatar.png`.
   - Implement real inference in `local_services/musetalk_server/app.py`
     (currently a stub emitting gray frames; `load()`/`render()` are marked
     `TODO[MuseTalk]`).
   - Set `AVATAR_PROVIDER=musetalk_local` in `.env`.
3. **Mandarin (zh-TW) swap (Phase 2)** — the research target. Set `LANGUAGE=zh`,
   `OPENROUTER_MODEL=qwen/...` or `deepseek/...`, a zh ElevenLabs voice; consider
   **Azure STT+TTS** (already wired) for better zh + Asia-region latency.
4. Optional: try **Gemini direct (Google API, Asia region)** to cut the LLM hop
   further than OpenRouter (US).

## Key files
- `pipeline/main.py` — pipeline assembly, LLM warmup, greeting, `/client`
  overlay injection.
- `pipeline/config.py` — all provider/model selection (driven by `.env`).
- `pipeline/stages/*.py` — per-stage factories (stt/llm/tts/avatar/vad).
- `pipeline/metrics.py` — `TtfoMeter` (logs `[TTFO]` per turn + summary).
- `local_services/` — local-model wrappers + servers (Phase 2/3).
- `scripts/preflight.py` — import/drift check; `scripts/bench_latency.py` —
  per-stage latency.
- Plan: `C:\Users\MARU\.claude\plans\i-come-to-taiwan-gleaming-sparrow.md`.
