# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **`STATUS.md` is the source of truth** for current state and what's in
> progress — read it first. **`WORKFLOW.md`** is the detailed end-to-end workflow
> (turn flow, avatar wire contract, running locally + remote, full `.env` reference).
> (The parent `E:\Claude\CLAUDE.md` describes a *different* repo, the `.claude` config
> workspace, and does not apply here.)

## What this is

A real-time **speech → STT → LLM → TTS → photoreal talking-head avatar** system.
Multi-turn, streaming end-to-end. Goal: time-to-first-output **< 3 s**. Built on
**Pipecat 1.3.0**, WebRTC to a browser at `/client`.

**Current stack (fully local TTS + avatar). See `STATUS.md` for the full state + the
A/V-sync architecture decision (read it before touching sync).**

| Stage | Service | Where |
|-------|---------|-------|
| VAD | Silero (local) | pipeline |
| STT | Deepgram nova-2 (`en-US`/`zh-TW`/`th` by `LANGUAGE`) default; **local OFFLINE alt `STT_PROVIDER=sherpa`** (sherpa-onnx streaming zipformer, bilingual zh-en, in-process CPU/~0 VRAM, zh→Traditional via OpenCC) | cloud / **local CPU** |
| LLM | `LLM_PROVIDER=openrouter` — OpenAI-compatible, so **cloud OR local Ollama** by `OPENROUTER_BASE_URL` (any model via `OPENROUTER_MODEL`); or `weather_chain` (Chinese weather bot) | cloud / local / remote |
| TTS | **CosyVoice** local streaming on vLLM in WSL — **now IN THIS REPO at `tts/cosyvoice-server/`** (2026-07-14; it used to be a separate repo). `COSYVOICE_MODEL=v2` (CosyVoice2-0.5B, **what `.env` actually runs**) or `v3` (Fun-CosyVoice3-0.5B, +flow-TRT). CUDA graphs are **ON for all languages incl. zh** (2026-07-14, reverses P33 — see below). Thai only: `TTS_PROVIDER=jaitts` (`:8004`) | **`:8001`, WSL** |
| Avatar | **MuseTalk** local mouth-region talking-head (**switchable preset**: `nimbus` female / `leo` his face+voice — `AVATAR_PRESET`), **TensorRT render by default** (`MUSETALK_TRT=1`) | **`:8002`, `musetalk` conda env** |
| Config | **Web config panel** — edit `.env` + restart the pipeline from a browser | **`:7870` (`:8444` over Tailscale)** |

**TTS note:** CosyVoice runs its autoregressive LLM on **vLLM inside WSL Ubuntu**
(`cosyvllm` conda env on the Blackwell 5060 Ti) — this cut first-chunk latency ~3.4s→~1.1s, the root
cause of the avatar lip-lag. The pipeline reaches it via `COSYVOICE_URL` set to the **WSL IP**, NOT
`localhost` (WSL2's localhost relay buffers the streaming audio ~2s). That IP changes on
`wsl --shutdown`, so **`launch.ps1` auto-heals a stale `.env` value against the live `wsl hostname -I`
on every start** (`Sync-CosyVoiceUrl`, 2026-07-15); only manual non-launcher starts still need a
hand-update. Run it with
`bash /mnt/e/Claude/VisualLLm/tts/cosyvoice-server/run_vllm_server.sh` in WSL. The original Windows `tts`-env
PyTorch server is the fallback (set `COSYVOICE_URL=http://localhost:8001` + start it). Full build
notes + gotchas: the `project-visualllm-cosyvoice-vllm` memory.

**Chinese TTS fix (2026-07-02, baked into the cosyvoice repo — `docs/PROBLEMS-AND-FIXES.md` P18):** running
the LLM on vLLM had **dropped CosyVoice's repetition-aware sampling (RAS)**, so zh intermittently looped on
the silence token → a ~4s sentence became ~12s of dead silence (heard as "halting" speech; the avatar kept
moving through the silence). Fixed by **restoring RAS as a vLLM logits processor**
(`CosyVoice/cosyvoice/vllm/ras_logits_processor.py` + `top_p=0.8` in `llm.py`; vLLM's own
`repetition_penalty` CANNOT be used — it CUDA-asserts on the `prompt_embeds` input). Separately, the zh
voice was choppy vs en purely because of the **reference clip** (`zero_shot` clones its rhythm); the baseline
now uses the fluid **"pro" AI-assistant voice** (`CosyVoice/asset/pro_ref.wav`, default in `tts_engine.py`) →
zh ≈ English pacing. An optional zh pause-trimmer (`COSYVOICE_SILENCE_CAP_S`, `_squeeze_silence`) is **OFF by
default** (not needed with the pro voice). Swap voices via `COSYVOICE_PROMPT_WAV`/`COSYVOICE_PROMPT_TEXT`.

**Traditional-Chinese garble fix (2026-07-11, baked into the cosyvoice repo `bb43be1` — `docs/PROBLEMS-AND-FIXES.md` P43):**
CosyVoice's text frontend **garbles long Traditional zh input** — spoken output degrades into noise past ~10 chars, while the
SAME sentence in Simplified is flawless. This pipeline feeds Traditional (llama-4-scout output; sherpa STT → Traditional), so
it was hitting the weak path; first-piece splitting (short comma-bounded pieces) hid it under the ~10-char threshold, but any
longer comma-less clause garbled. Fix = **`COSYVOICE_T2S=1` (default on)** converts Traditional → Simplified with OpenCC
`t2s` **before** synthesis, in `tts_engine.py` (`_to_simplified()` at the top of both `synthesize`/`synthesize_stream`, covers
`/tts` + stream + warmup; no-op on en, and no-op if opencc is missing). **Inaudible** — T and S are the same spoken Mandarin.
Reproduced + verified locally (transcribe-back A/B): Traditional now synthesizes as cleanly as Simplified. `COSYVOICE_T2S=0`
disables. A distinct root cause from the RAS silence-loop (P18) above.

**vLLM CUDA graphs — GRAPHS ON EVERYWHERE, zh INCLUDED (2026-07-14; this REVERSES P33).**
`COSYVOICE_VLLM_EAGER` default is **`0`** (graphs on) in `run_vllm_server.sh`, and that is the correct setting for
**every** language. The live baseline runs `COSYVOICE_MODEL=v2` + `LANGUAGE=zh` + graphs ON, and **the user's live eye
says the Chinese lipsync is fine.** Graphs are also the TTS-first-chunk win (avg ~2.0→~0.85s, P27). So: **do not flip
`COSYVOICE_VLLM_EAGER` back to `1`, and do not "fix" the live config to eager on the strength of P33** — that verdict
was wrong. The config panel's **CUDA graphs** card toggles it (rewrites the script + relaunches WSL) if you ever need to.

**Why P33 said the opposite, and why it was wrong** (`docs/PROBLEMS-AND-FIXES.md` P27/P31/P32/P33): P31 reverted graphs
for "live inconsistency"; P32 measured the TTS side (`tts/cosyvoice-server/_ttfb_variance.py`) and found graphs
*faster + lower-variance* than eager even under real MuseTalk render (96 samp: graphs 1.29/2.23/0.37s vs eager
1.94/3.43/0.64), so P31's "shape-spike" mechanism did not reproduce. P33 then measured the zh **audio**
(`_zh_audio_ab.py`) and found graphs ON does slightly alter the zh waveform (longer, more internal silence) — the graph
decode perturbs the zh-critical **RAS** sampling (the P18 fix) — and *inferred* that, since MuseTalk lip-syncs off a
**Whisper of the waveform**, this must degrade zh lipsync. **That inference never held up live.** The audio delta is real
but too small to reach the eye. **The lesson cuts both ways:** the house rule is "the probe passes what the eye rejects"
(P19) — P33 ran it backwards, turning a *measured difference* into a *predicted defect* the eye never confirmed. A
measurable delta is not automatically a perceived one; the live eye is the arbiter in **both** directions.
The independent Lever-4 poll-tighten (`model.py` 0.1→0.02) stays.

**Shared-GPU VRAM (why "won't talk" can mean CosyVoice crashed):** vLLM and MuseTalk share the one
16GB card. vLLM's `gpu_memory_utilization` (env `COSYVOICE_VLLM_GPU_UTIL`, **default `0.07` since
2026-07-15**, set in `run_vllm_server.sh` — NOT `.env`; the launcher forwards only `COSYVOICE_MODEL`)
must clear vLLM's non-KV floor (~0.98GiB: weights 0.7 + CUDA graphs 0.15 + activations) plus a KV
cushion, or load crashes with "No available memory for the cache blocks". With `COSYVOICE_VLLM_MAX_LEN`
capped at 2048 the KV need is tiny, so 0.07 (KV 0.16GiB = ~6.7 max-len seqs, ~35x a real turn) is
verified clean (24s zh paragraph, /tts + /tts/stream, gen speed unchanged; whole WSL server
3.8GB→2.3GB; the wall is ~0.062 on 16GB — if a vLLM/driver bump fails to load, raise to 0.08+).
The old "~4GB footprint / 0.3 default" text described the UNCAPPED max_len era. If the avatar shows but
the bot is silent, first check `:8001` is actually up — the pipeline log shows "Cannot connect to host
…:8001". The "Available KV cache memory" log line must be positive.
**LOAD ORDER STILL MATTERS: start CosyVoice (vLLM) BEFORE MuseTalk** (the low util makes vLLM far
friendlier to a busy card, but keep the order). Clean recovery = stop all three → start cosyvoice
(`run_vllm_server.sh`) → then `scripts/run.ps1` (MuseTalk + pipeline). The launcher already does this
order. (`docs/PROBLEMS-AND-FIXES.md` P15.)

**Chinese first-chunk is slower than English, and each language has its own TTFO lever (updated
2026-07-04 — the 2026-07-03 hop_zh=5 verdict is REVERSED, see P22).** CosyVoice's first-chunk TTFB
scales with the INPUT sentence length (it prefills the whole sentence before the first audio token). The levers:
- **`COSYVOICE_FIRST_HOP_ZH=0` (baseline since 2026-07-04, default `:-0` in the cosyvoice repo's
  `run_vllm_server.sh`).** hop=5 (a smaller opening TTS chunk) was the 2026-07-03 zh lever, but a live A/B
  proved it HURTS live zh TTFO: the small chunk fills the `MUSETALK_LEAD_FRAMES=14` synced-start cushion
  slowly → the steady-hold balloons (zh hold ~1.9–2.2s vs en ~0.85s; the entire zh-vs-en TTFO gap). hop 5→0
  cut zh median **4.14→3.09s**, screen clean, lips-start *improved*. Its isolated-TTFB win never survives the
  synced-start fill (P19's caveat, now resolved: `docs/PROBLEMS-AND-FIXES.md` P22). English was always hop=0
  (`COSYVOICE_FIRST_HOP_EN`; the old global hop=5 pushed en lip-start ~0.70→~1.95s).
- **`COSYVOICE_FIRST_PIECE` (the first-clause split, `.env`) = the en lever.** en's long sentences benefit
  from starting speech on the first *clause* early. Splits at ASCII comma/space past MIN/MAX char thresholds.
- **`COSYVOICE_FIRST_PIECE_ZH=1` (2026-07-04) = the zh lever** (`docs/PROBLEMS-AND-FIXES.md` P23). The en
  split never fires for zh (ASCII comma/space vs zh's full-width ，and no spaces), so a long zh opener — the
  LLM ignores the ≤10-char-opener prompt rule on ~30% of turns — still prefilled whole (TTS TTFB ~3.1s, turn
  ~4.8s). The zh path flushes the turn's first piece at a full-width **，；： ONLY, never a char cap** (a cap
  cuts mid-word — 天氣預|報 — the rejected splitter; a comma boundary cannot), guarded by
  `COSYVOICE_FIRST_PIECE_ZH_MIN_CHARS=5` CJK-counted chars (the opening piece's audio must cover the next
  piece's synthesis or the voice pauses between clauses). Live A/B: long-opener turns **4.78→3.08s**,
  split-fired audio gaps 59–65ms (no pause); comma-less zh + en byte-identical.
  (`local_services/first_piece_aggregator.py`; knobs read via `os.getenv` inside the aggregator.)

**zh turn-start "breathing sound" — FIXED 2026-07-14 (`docs/PROBLEMS-AND-FIXES.md` P34).** The breath is now trimmed
**server-side inside CosyVoice** (`tts_engine.py::_trim_lead_in`, cosyvoice `290d17e`; `COSYVOICE_TRIM_LEAD=1` default,
`=0` reverts) — lead-in before the first audible word **0.23s median → a deterministic 0.03s**. Detection is **frame
RMS, not per-sample abs** (a breath spikes above any sample threshold while staying inaudible). The CLIENT-side trim
described below stays REJECTED — whole tensors server-side is the only safe place. Paired with the LLM Groq pin
(`OPENROUTER_PROVIDER_ONLY=Groq`, LLM TTFB 1.26→0.94s and its variance gone), a real-browser turn reaches the ear in
**2.74–3.09s on a fresh session** (was 3.69s). **CAVEAT — the session-degradation bug is now the dominant TTFO cost:**
after ~3 long turns the avatar's turn-start silently gains ~1.0s (`lips start +0.48s → +1.47s`) and never recovers, so
turn 6 lands at 4.37s. Suspected backlog from very long replies (the bot emits 35–38s of audio for a "one short
sentence" question). UNRESOLVED — measure a fresh session, not a long one, or you will mis-attribute.

**Historical (2026-07-05) — the original no-trim writeup:** CosyVoice's zero-shot synth prepends a low-level breath (25–610ms, −34..−68 dB) before the first word on
~every zh piece; the avatar lip-syncs off a Whisper of the waveform so the mouth moves over it ~0.3–0.6s before the
answer. A start-of-turn byte-stream trim was tried and **REJECTED** (crashed the first piece on aiohttp's odd-sized
chunks → "only speaks one sentence per turn"; user judged no-trim better). The breath is accepted as baseline; any
re-attempt must trim **server-side in CosyVoice** (whole buffers), not in the pipecat client.

`MUSETALK_LEAD_FRAMES` below 14 is a **CLOSED question (2026-07-04): REJECTED by the user's live eye.**
lead=8 measured zh 3.03/en 2.48 median at hop=0 (the first all-under-3s config, probe-screen clean), but the
user live-tested every value below 14 and saw delay or avatar freezes — the probe screen misses what the eye
catches (P19's lesson, twice now). Don't re-try lower leads. The remaining TTFO levers are the TTS first-chunk
cost, the P20 shared-GPU collision (stagger / stream-priority, untried), and the structural fix: a dedicated
avatar GPU. NOTE: lead reaches the avatar server only via a full relaunch (launcher/`run.ps1`) — the config
panel's Restart cycles the pipeline only, so a panel-edited lead never takes effect.

Each stage is a thin single-provider factory in `pipeline/stages/` chosen by `.env` — these
are **deliberate fallback switches, not multi-provider branching**:
- `TTS_PROVIDER` = `cosyvoice` (default) | `jaitts` (the local Thai voice, `:8004` — CosyVoice cannot
  speak Thai). Anything else now **raises** instead of silently falling through to a cloud voice.
  (The `moss` / `elevenlabs` / `deepgram` branches were removed 2026-07-14 — never once selected. In git history.)
- `LLM_PROVIDER` = `openrouter` (default; point `OPENROUTER_BASE_URL` at `https://openrouter.ai/api/v1` for
  cloud or `http://localhost:11434/v1` for a local Ollama model) | `weather_chain` (NCU zh weather bot).
  **`OPENROUTER_PROVIDER_ONLY` (2026-07-04, TTFO lever):** pin OpenRouter to a fast backend (default
  **`Groq`**) instead of the default transpacific Gemini route — the LLM hop was the dominant TTFO cost +
  all its variance. Injected as `extra_body.provider.only` via pipecat's `Settings.extra` (`stages/llm.py`).
  Cut the LLM hop ~1.1–1.6s (tail to 3.6s) → **~0.7s tight** (zh 1.64→0.80s median, en 1.07→0.67s). Empty =
  unpinned Gemini, fully revertible. End-to-end TTFO only modestly down (TTS + steady-hold now dominate); the
  real prize is the killed 7–8s LLM-tail. **Model baseline = `meta-llama/llama-4-scout`** (Groq, non-reasoning):
  same speed as `llama-3.3-70b`, clean *substantive* Traditional zh, and ~5× cheaper ($0.11/$0.34 vs the 70b's
  real Groq price $0.59/$0.79). Rejected: `llama-3.1-8b` (zh errors), all mid-cost models (reasoning → slower).
  Judge model quality with an ISOLATED probe. (**Correction, P45:** `pipeline.log` DOES log each turn's committed reply
  text — the `[commit-dbg]` line in `_TranscriptStore.add`. The old "it never logs the reply text" claim here was false,
  and believing it is what hid P45's transcript corruption for weeks. **Keep that log line.** But note the text it
  logged BEFORE P45 was damaged, so don't judge past model quality from old log lines either.) `docs/PROBLEMS-AND-FIXES.md` P21.

**The web config panel (`local_services/config_panel/`, `:7870`) is the easy way to change all of this**
— it edits `.env` in place (preserving comments) and restarts the pipeline. Run it with the system
Python: `python -m local_services.config_panel.server`. Its Restart kills `:7860` via a native Win32
`TerminateProcess` (NOT `taskkill`/PowerShell — those hang for tens of seconds under CPU load here). It also
hosts the **"Avatar preset"** card (swap the whole avatar identity — Nimbus female / Leo his-face-and-voice; see the
`/studio/` + presets note under Architecture) alongside the CosyVoice-model, CUDA-graphs, and Avatar-output cards.
It also exposes the **VAD** knobs (2026-07-14) — `VAD_STOP_SECS` (curated) + `VAD_START_SECS`/`VAD_CONFIDENCE`/
`VAD_MIN_VOLUME` (advanced). These were **hardcoded** in `stages/vad.py` (the one stage the panel couldn't reach);
they are now `.env`-driven like everything else, with the old values as defaults, and `build_vad_params()` **logs the
live values** (`VAD: stop=… start=… …`) so you can confirm a panel edit took effect. **Scope, so the knob isn't
over-trusted:** under the baseline `ALLOW_INTERRUPTIONS=1`, `main.py` passes NO `user_turn_strategies`, so pipecat's
defaults apply and end-of-**turn** is called by `TurnAnalyzerUserTurnStopStrategy` (**Smart Turn v3**, semantic) — the
VAD only supplies the speech segmentation it runs on. So `VAD_STOP_SECS` *shapes* responsiveness, it does not dictate
it. And none of it ever appears in TTFO, whose **t0 IS the turn-end** — it is latency the user feels but the metric
cannot see. (Log quirk: on the FIRST connection of a process that `VAD:` line goes to stdout only — the file sink
attaches later — so look for it on a subsequent connection.)
**The pre-t0 lever that ACTUALLY paid (2026-07-16, P54) is `ttfs_p99_latency`, NOT the VAD.** After Smart Turn returns
`COMPLETE`, the strategy still waits `ttfs_p99_latency - stop_secs` for the STT's final transcript (it short-circuits
only on `TranscriptionFrame.finalized=True`, which sherpa's/Deepgram's streaming path never sets). Sherpa declared no
value → pipecat's **1.0s cloud-STT default** (`ttfs_p99_latency not set, using default 1.0s` in the log) → a flat **1.0s
of dead wait every turn**, though sherpa emits its final transcript *synchronously with the endpoint* (already in hand).
Fixed by `kwargs.setdefault("ttfs_p99_latency", 0.1)` in `sherpa_stt.py` (non-zero avoids the `stop_secs>=ttfs` collapse
path): `COMPLETE→t0` **1.0s → 0.09s**; content unaffected (the wait is entirely POST-transcript, so it delays only WHEN
the turn fires, never WHAT it contains). If you swap STT, declare its REAL measured value — the default is a guess about
someone else's service, billed to you every turn. `docs/PROBLEMS-AND-FIXES.md` P54 · matrix: `docs/LATENCY-MATRIX.md`.

**Removed 2026-07-14 (dead-code audit): the `moss` / `elevenlabs` / `deepgram` TTS branches and the
`funasr` STT branch.** None had ever been selected -- the stack has always been Deepgram STT ->
OpenRouter LLM -> CosyVoice TTS -> MuseTalk. An untried fallback is not a safety net; `git revert`
restores any of them. A typo'd `TTS_PROVIDER` now RAISES instead of silently falling through to
ElevenLabs (a cloud voice, and a bill).

Core `.env` knobs: `LANGUAGE` (en/zh/th), `TTFO_TARGET_SECONDS`, `TTS_PROVIDER`,
`VAD_STOP_SECS`/`VAD_START_SECS`/`VAD_CONFIDENCE`/`VAD_MIN_VOLUME` (**0.5/0.2/0.7/0.6**, Silero; panel-editable —
see the config-panel note above for what they do and do NOT control),
`COSYVOICE_TRIM_LEAD` (**1 = default, in the cosyvoice repo**: strips the inaudible leading breath CosyVoice
prepends to every zh piece — 0.23s median, up to 0.60s of pure TTFO dead time. Server-side, on whole tensors, which is
the ONLY safe place (the client-side byte-stream trim crashed — P34). `0` reverts. **Wants an eye check:** MuseTalk
lip-syncs off a Whisper of the waveform, so this changes what Whisper sees at turn start; the probes cannot judge the
first viseme),
`COSYVOICE_MODEL` (**`v3`** = current baseline, Fun-CosyVoice3-0.5B, +flow-decoder TRT + CUDA graphs; `v2` =
CosyVoice2-0.5B. Read by `run_vllm_server.sh` + the launcher; the v3 block also defaults `COSYVOICE_FLOW_TRT=1`.
Switchable in the config panel's **CosyVoice model** card, which relaunches the WSL TTS server — a plain `.env` edit
needs that relaunch, NOT just the pipeline Restart. `docs/superpowers/plans/2026-07-10-cosyvoice3-ab.md`),
`MUSETALK_SYNC_MODE` (**`steady`** = video-master, synced start, the user's pick and current
default; `live` = audio-master, voice instant + lips trail ~0.75s, can never pause. The old
**steady "screech" is FIXED** — it was pipecat discarding the partial audio buffer after a >3s
render-stall gap (`BOT_VAD_STOP_FALLBACK_SECS`); see `docs/PROBLEMS-AND-FIXES.md` P3 +
`main.py::_relax_bot_vad_stop_timeout` and the producer-side sample alignment in
`cosyvoice_tts.py::run_tts` (P52 — replaced the old `_align_even` consumer patch). Remaining steady
tradeoff: under a long render stall the voice briefly **pauses** then resumes clean — switch to
`live` if that pause is worse than the lip trail), `MUSETALK_FPS` (**14** now (the user's pick); a divisor
of 16000 (8/10/16/20/25) makes frame count = audio length exactly, but the server's `samples_for_frames`
ceil sizing makes a non-divisor like 14 correct anyway; the old `int(16000/fps)` truncation lost ~1 frame/segment → lips
finished ~1–2s early, `docs/PROBLEMS-AND-FIXES.md` P9. NOTE: the end-of-turn leftover-audio blip
(P10) is **FIXED** — `int()`→`math.ceil` on the `audio_cap` in `musetalk_video.py::_advance` so the
final audio sub-frame releases in step instead of waiting for the delayed `video_end` drain),
`MUSETALK_FEED_BURST_S` (1.0 — bursts the first 1s of a turn's audio un-paced so the renderer
isn't starved at turn start; cut lip-start lag ~1.9s→~0.8s), `MUSETALK_END_TAIL_FRAMES` (**0** now —
the client close-crossfade replaces the neutral tail; `>0` = static neutral frames after speech, the
old clean snap), `MUSETALK_CLOSE_FADE_FRAMES` (**5** — eases the mouth shut at end of turn: the client
cross-dissolves the last spoken frame→rest pose over N frames, delivered **free-run/untagged** ("live
during the close") so it survives steady's non-live transport without the audio-cap stranding it; `0`
= clean snap; needs `END_TAIL=0`; `docs/PROBLEMS-AND-FIXES.md` P12), `MUSETALK_IDLE_MOTION` (**0** = no breathing idle; the face
holds the static neutral portrait between turns — the user's pick. `1` = the synthesized breathing
loop. Server reads it from the OS env, so `run.ps1` propagates it), `MUSETALK_SIZE` (**512** now — the delivered
frame px, couples server+client+`video_out`. Bumped from 256 for a crisper studio/hair (the model still generates
the face at a fixed 256px, so higher res only sharpens the STATIC frame, not the animated mouth). **512 is the
lag-free ceiling on this shared GPU: 768/1024 profiled with render headroom in ISOLATION but dropped to ~10fps under
live CosyVoice GPU contention → steady-mode voice lag** (`docs/PROBLEMS-AND-FIXES.md` P36); higher res needs a
dedicated avatar GPU. Also pair with `MUSETALK_BASE_MAX` (source-portrait res cap, **768**; higher = sharper
background but heavier composite) and keep `MUSETALK_FPS` identical across server+pipeline or you get drift),
`MUSETALK_SPLIT` (**0 = default, full-frame**; `1` = stream ONLY a fixed-size mouth crop and let `/studio`
composite it over a pristine, never-video-compressed background still → crisp picture, VP8 budget concentrated
on the mouth [`/nimbus` removed 2026-07-14; `/studio` is the surviving custom client]. **`/studio` ONLY** — the prebuilt `/client` can't composite and is unsupported while on (it would
show a floating crop); `/client` stays the untouched full-frame fallback at `=0`. **The BACKGROUND gets genuinely
sharp; the animated mouth stays MuseTalk's 256px** (a model limit, not transport — removing VP8 blur helps but it
never goes photo-crisp). A/V sync is REUSED verbatim (the sync path is frame-content-agnostic — it just pins frame
N to audio N/fps, so a small crop changes nothing). Targets `MUSETALK_IDLE_MOTION=0` / a STATIC portrait (a moving
bbox would break the fixed-crop mapping; split mode force-disables the idle loop). Automatic for any `AVATAR_REF` —
the server derives the bbox+background from its existing one-time preparation. Server: `GET /overlay-assets`
(background PNG + bbox); pipeline proxy: `GET /client/avatar-overlay`; `/studio` canvas-composites. **Toggle it in
the config panel's "Avatar output" card** (`:7870`/`:8444`) — unlike a plain `.env` edit, that card relaunches the
**avatar server AND the pipeline** (`restart_avatar()`), because a plain edit + the panel's pipeline-only Restart would
resize the track to 256 while the server still streams 512 → a broken mismatch. Verified
offline (seamless composite, sharper bg A/B) + live in a real browser (WebRTC track = 256², canvas paints the 768
bg, no seam). `MUSETALK_SPLIT_SIZE` (**256** — the square px of the streamed crop; MUST match server + pipeline,
un-stretched into the bbox rect client-side). `docs/superpowers/specs/2026-07-11-mouth-crop-overlay-design.md` +
`docs/superpowers/plans/2026-07-11-mouth-crop-overlay.md`),
`MUSETALK_TRT` (**1 = default, load-bearing for A/V sync**: TensorRT UNet+VAE render path,
per-segment render ~389ms→~255ms so the avatar keeps ~12fps under CosyVoice's shared-GPU
contention — where the PyTorch path drifts seconds behind the voice on long turns. Engines live in
`musetalk_server/trt_cache/` (~1.75GB, gitignored, GPU/driver-specific — rebuild with `trt_build.py`);
any load failure silently falls back to PyTorch. `0` = PyTorch. `docs/PROBLEMS-AND-FIXES.md` P16.
**`MUSETALK_FREE_TORCH` (1 = default, 2026-07-15 VRAM trim):** once the engines load, the torch
UNet+VAE are dropped (−1.8GB; avatar server ~5.2→~3.3GB) — the fallback decision already happened at
load time, so they were pure dead weight. `0` keeps them resident. Verified: `_drive_frames` renders
audio×fps ±1 after the free. Gotcha that cost the first attempt: `_free_torch_render_models` runs
inside `load()`, whose LOCALS still alias the wrappers — null the INNER attrs (`unet.model`,
`vae.vae`), not just `self.unet`, or nothing frees),
`MUSETALK_GPU_COMPOSITE` (**1** — runs the per-frame mask-blend + downscale on the GPU (torch) instead
of CPU PIL/cv2: composite ~73ms→~11ms per 8-frame seg → total render 246→182ms (−26%, ceiling ~33→44fps).
**Only active with `MUSETALK_TRT=1`** (the VAE output is already a GPU tensor there; the PyTorch path
keeps the CPU composite). Output is pixel-identical — SSIM 1.0, ≤1 LSB vs the CPU path. Falls back to CPU
if a crop_box runs off-frame. Code default off (opt-in); `app.py::_composite_gpu`. **Benchmarked: at 12fps
it does NOT reduce A/V drift** (TRT already holds render ≥12fps, even under 100% GPU contention) — the win
is reserve headroom + a freed CPU, judged by the live call. `docs/PROBLEMS-AND-FIXES.md` P17),
`MUSETALK_LEAD_FRAMES` (**14, load-bearing, CLOSED at 14** — it IS the synced-start delay AND a mid-turn
shock absorber; lower starves the queue → freeze. The 2026-07-03 sweep rejected lowering it (P19), and on
2026-07-04 the user live-eye tested **every value below 14** and saw delay or avatar freezes — even `lead=8`
at hop=0, which had measured zh 3.03/en 2.48s median probe-screen clean. The probe misses what the eye
catches; do not re-try lower leads. Server-side knob: only a full relaunch applies it, not the panel Restart),
`COSYVOICE_PACE_RATE` (**UNSET = OFF, verified 2026-07-14 in the running server's own `/proc/<pid>/environ`** —
this doc used to claim "1.3"; it is not set in `.env`, not in `run_vllm_server.sh`, and the code default in
`tts/cosyvoice-server/app.py` is `0` = no pacing. **CosyVoice already streams unthrottled.** When it IS set it caps
the OPENING chunks to `rate`x real-time so the TTS GPU burst doesn't collide with MuseTalk's first render segment;
it stops after `COSYVOICE_PACE_WINDOW_S` (2.0s) of emitted audio, and it **never delays the first chunk** — so it is
NOT a TTFO lever in either direction), `COSYVOICE_FIRST_PIECE` (**1 = default, TTFO win**: emit a short opening CLAUSE to
TTS first, then normal sentences. CosyVoice's first-chunk TTFB scales with the INPUT sentence length
— it prefills the whole sentence before the first audio token, so a 16-word opener costs ~3.0s vs
~1.7s short. Splitting cut TTS first-chunk ~3.0s→~1.7s and **TTFO ~4.6s→~3.2s**, flow stays smooth
(delivered audio gap ~55ms, never a stall). `COSYVOICE_FIRST_PIECE_MIN_CHARS`/`_MAX_CHARS`
(**18/32** = tuned sweet spot: use an early comma if present, else cap at ~32 chars on a word
boundary — enough opening audio to cover the next piece's synthesis even under shared-GPU
contention; smaller MAX = faster start but risks a between-clause pause). `0` = off.
`local_services/first_piece_aggregator.py`), `COSYVOICE_FIRST_PIECE_ZH` (**1 = the zh split,
2026-07-04**: the en split above never fires on Chinese — full-width ，vs ASCII comma, no spaces —
so long zh openers cost ~3.1s TTFB; this flushes the first piece at a full-width ，；： ONLY, never
a char cap, min `COSYVOICE_FIRST_PIECE_ZH_MIN_CHARS`=5 CJK chars. Long-opener turns 4.78→3.08s, no
between-clause pause. P23), `FILLER_WORDS` (**live `.env` = 0 (OFF) since the zh P30 caveat below; the 2026-07-05
"1 = baseline" claim is stale — the knob remains available**: `1` OPENS the turn on a
rotated natural "thinking" phrase ("嗯，讓我想一下喔，…") synthesized through the normal TTS path — one
continuous turn, zero screech (audio gap ~60ms) — so the avatar starts talking + lip-moving ~0.7s sooner
(zh def 2.91→2.23s, wx 2.38→2.03s). **HONEST: a PERCEPTION win, not a speedup — TTFO counts time-to-first-SOUND
and that sound is the filler; the real ANSWER arrives slightly LATER (queued behind it).** **zh CAVEAT (2026-07-05, P30): the filler makes zh feel DELAYED** — the avatar starts on the filler, then the
real zh answer (chopped into short comma/sentence pieces, each ~0.8s TTFB) lags behind, and each short piece's
audio barely covers the next piece's synth → micro-gaps read as "avatar first, voice delayed." en escapes it
(longer fillers + char-count splitting = bigger pieces). Fix = `FILLER_WORDS=0` (confirmed smooth in zh; costs
~0.7s TTFO) or make the filler en-only. NOT the raw TTS speed — zh TTFB ≈ en (~0.9s) once graphs were off.
Fillers are ~1.2s
each so the first piece fills the `MUSETALK_LEAD_FRAMES` cushion — a too-short "嗯，" ballooned the hold 0.5→1.7s
(the fix). `FILLER_WORDS_COUNT` (**1**) chains more for a longer opener. Needs `COSYVOICE_FIRST_PIECE=1` (shares
that aggregator); `0` = off. `local_services/first_piece_aggregator.py`, P26), `CLIENT_FORCE_SPEAKER` (**1 = default**: phone browsers play the voice
on the LOUDSPEAKER, not the earpiece — Android Chrome flips to ear-style 'communication' routing
while the mic is live; iOS gets a WebAudio fallback. Mobile-UA only, desktop/headphones untouched;
the phone self-reports to `[speaker-debug]` in pipeline.log. P24),
`CLIENT_JITTER_BUFFER_MS` (raise only for a remote/WAN viewer),
`WEBRTC_VIDEO_BITRATE_MAX` (caps aiortc's VP8 ceiling so the video fits a WAN link), and
`WEBRTC_ICE_SUBNET` (**`100.64.0.0/10`** = pin WebRTC ICE to the Tailscale interface; fixes the
intermittent remote mic — `0` disables), and — for a **PUBLIC link anyone can use** (no Tailscale).
The public **front door is a Cloudflare quick tunnel** (2026-07-11; `scripts/tunnel.ps1`, auto-started
by `launch.ps1` step [4/5], prints a random `https://<random>.trycloudflare.com` URL) — it replaced
Tailscale Funnel (which stopped working: a login-gated admin toggle, and `funnel reset` wipes serve).
Like Funnel, the tunnel carries only the page + `/api/offer` signaling, NEVER the WebRTC media, so the
media is enabled separately by `WEBRTC_PUBLIC` (**now `1` = baseline**; `0` = tailnet-only. `1` advertises
STUN so an off-tailnet browser reaches the media. This box's NAT is port-preserving (cone), so STUN-only
reaches many networks. When on, `_restrict_ice_to_subnet` keeps a SET = {Tailscale 100.64/10 for
tailnet/same-LAN pairs + the internet-facing default-route `/32` for the public srflx}, dropping
hyper-v/radmin noise — pinning to EITHER one alone breaks the other's clients). For a visitor whose own
network is **symmetric-NAT / UDP-restricted** (STUN can't punch it — the "page loads but avatar stuck on
connecting" symptom), a TURN **relay** is required: **`TURN_CLOUDFLARE`** (**default ON when
`WEBRTC_PUBLIC=1` and no static `TURN_URLS`; `0` opts out**) fetches a FRESH zero-signup relay per
connection from Cloudflare's speed-test endpoint (`speed.cloudflare.com/turn-creds`, `main.py::_cloudflare_turn`,
5-min cache, silent STUN-only fallback) — verified live: a relay-only client connects over `turns:5349`
(firewall-proof). `TURN_URLS`/`TURN_USERNAME`/`TURN_CREDENTIAL` (a static relay — e.g. an official free
Cloudflare Realtime TURN key, same `turn.cloudflare.com` servers) override the Cloudflare-fetch path.
All gate `_install_turn_ice_servers` (server-side ICE-server injection + `/client` head-patch +
`/client/ice-config` for `/studio`). Still single-client + unauth. `docs/PROBLEMS-AND-FIXES.md`
P38. **Full reference: `WORKFLOW.md` §8.**

## Commands

There is **no build/lint/unit-test suite** — don't invent one. The real commands (3 processes;
`scripts/run.ps1` starts the avatar server + pipeline and propagates the MuseTalk env from `.env`):

**One-click full stack (easiest):** double-click **`Run VisualLLm.exe`** in the repo root. It runs
`scripts/launch.ps1`, which brings up the WSL CosyVoice TTS (waits on `/health`), then `run.ps1`
(avatar + pipeline), then the config panel, then opens `/client/`. The launcher window is the
on/off switch — press Enter (or close it) to stop everything. The `.exe` is a tiny C# shim compiled
from `scripts/Launcher.cs` by the bundled `csc.exe`; rebuild it with `.\scripts\build-exe.ps1` (only
needed if you change `Launcher.cs` — editing `launch.ps1` needs no rebuild). The individual commands
below are still the way to run/debug a single stage.

```bash
# 1. CosyVoice TTS server (DEFAULT = vLLM in WSL, TTFB ~1.1s) -- NOW IN THIS REPO at tts/cosyvoice-server/
wsl -d Ubuntu -e bash -c "bash /mnt/e/Claude/VisualLLm/tts/cosyvoice-server/run_vllm_server.sh"   # serves :8001 in WSL
#    Then set .env COSYVOICE_URL to the WSL IP (NOT localhost — WSL2 relay buffers the stream): `wsl hostname -I`.
#    (launch.ps1 heals a stale WSL IP automatically on every start — Sync-CosyVoiceUrl; manual starts don't.)
#    FALLBACK = Windows PyTorch server (slower, TTFB ~3.4s), set COSYVOICE_URL=http://localhost:8001 :
#      E:\miniconda3\envs\tts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8001
#    (COSYVOICE_PACE_RATE is UNSET -> code default 0 = pacing OFF; the Windows server needs SSL_CERT_FILE=<certifi> —
#     the tts/musetalk conda envs have a broken Windows cert store; see STATUS.md/memory.)

# 2. MuseTalk avatar server — `musetalk` conda env (NOT the pipeline env), serves :8002
E:\miniconda3\envs\musetalk\python.exe -u -m local_services.musetalk_server.app
#    (reads AVATAR_REF / MUSETALK_SIZE / MUSETALK_FPS from the OS env ONLY — no python-dotenv)

# 3. Pipeline — project main env (SYSTEM Python 3.11, has pipecat — NOT a conda env); serves /client + /studio
python -m pipeline.main            # prebuilt: localhost:7860/client/  |  custom Studio UI: localhost:7860/studio/

# --- or start the avatar server + pipeline together ---
.\scripts\run.ps1

# 4. (optional, Thai only) JaiTTS-F5TTS server -- shared F5 venv, serves :8004 (TTS_PROVIDER=jaitts).
#    python -m uvicorn local_services.jaitts_server.app:app --host 0.0.0.0 --port 8004

# 5. (optional) Web config panel -- SYSTEM python (it restarts the pipeline), serves :7870.
python -m local_services.config_panel.server               # edit .env + restart from the browser

# Verify every fragile import resolves WITHOUT keys/network (Pipecat drift check):
python -m scripts.preflight

# Avatar A/V test tooling (close any /studio tab first — server is single-client):
# UNIFIED mic-to-ear latency harness (PREFER THIS). Now a PACKAGE at scripts/measure/ (entry
# `python -m scripts.measure` unchanged): logparse.py (per-turn anchors), waterfall.py (stage
# table + median/p95 + fresh/warm), drive.py (probe + Playwright), report.py (writers + history),
# __main__.py (orchestration). Drives N turns, writes output/measure_report.json + docs/measure_data.js
# (docs/workflow-timeline.html auto-uses it) + appends output/measure_history.jsonl.
python -m scripts.measure --turns 5                                # real Chromium (true browser E+F) + fallback probe
python -m scripts.measure --observe --turns 5                      # DON'T drive: parse the last N turns YOU just spoke
python -m scripts.measure --no-browser --turns 3 --offline-capture # headless probe: precise capture + arrival, est playout
python -m scripts.measure --compare -2 -1                          # diff the last two history runs (did a change help?)
#   PRE-t0 vs POST-t0: TTFO's stopwatch STARTS at t0 (user-stopped), so it can NEVER see the cost of
#   DECIDING t0. Felt delay = pre-t0 + post-t0. That blind spot hid a full second (P54). The measured
#   real-turn matrix + what-to-attack-next lives in `docs/LATENCY-MATRIX.md` (median mic-to-ear 2.91s).
#   *** NEXT TARGET + THE ONLY ROW LEFT WITH HEADROOM: `docs/TTS-FIRST-CHUNK-HANDOFF.md` (TTS 0.93s). ***
#   THE HARNESS LIES -- read `docs/PROBLEMS-AND-FIXES.md` P55 BEFORE trusting any row. It invented ~0.2s of
#   "network" that was never on the wire (transport is CLOSED: ~0.13s real, at its floor -- and its listed
#   lever WEBRTC_VIDEO_BITRATE_MAX caps VP8 VIDEO while the voice is a separate OPUS track, so it could never
#   have moved it). Two traps that survive: `--btail` MUST exceed the bot's reply (~50s! the 32s default makes
#   every turn interrupt the last -> the render row inflates and fakes a "session degradation" bug), and
#   `--blead 2` < the ~5s ICE handshake so the driver's FIRST turn is LOST every run ("drove N, got N-1").
#   The tool now warns on both instead of silently backfilling strays from older sessions.
#   RULE THAT CAUGHT ALL 5 FALSE FINDINGS: check a row against its PHYSICS FLOOR before optimising it;
#   13x over budget is the instrument, not the system. Then instrument the seam -- never argue.
#   JUDGE PRE-t0 ONLY FROM `--observe` ON REAL SPEECH: the synthetic clip's comma pause makes Smart Turn
#   re-poll (it re-runs once per VAD pause) and INVENTS a ~1.6s pre-t0 cost real speech does not have
#   (5 real turns: 0 INCOMPLETE polls). `--observe` leaves Capture blank on purpose (human speech length
#   unknown -- it refuses to fake it) and prints the Smart-Turn verdict trace instead.
#   The ~11-stage WATERFALL runs from the true MIC moment to the user's EAR, each row tagged with its
#   SOURCE and the .env LEVER that moves it:
#     Capture (speech-end -> t0: VAD hangover + Smart-Turn end-of-turn)  [driver]  <- VAD_STOP_SECS
#       = (t0 - 'User started speaking') - the wav's energetic speech length. PRE-t0 latency the TTFO
#         metric can't see; log-derived, works on both paths. NOT from a pre-connect clock (ICE+greeting skew).
#     STT-finalize | LLM TTFB | LLM->TTS flush | TTS TTFB | Avatar render | steady lead-hold   [log]
#     Transport+encode+network  [probe = headless arrival]
#     Browser jitter buffer [browser-stats = getStats] | Browser decode+playout [browser-audio = WebAudio onset]
#   REAL browser output delay via TWO paths (the 2026-07-14 removal of the /client beacon is REVERSED, done right):
#     * auto: `--turns N` drives a real headless Chromium (Playwright) with the wav as a fake mic on
#       /studio/?measure=1; its beacon POSTs jitter/onset -> pipeline.log `[client-playout]` line.
#     * human: open /studio/?measure=1 yourself (incl. remote); the same beacon fires (min-RTT clock sync).
#     Both land on the pipeline clock; `--no-browser` falls back to the probe (F row becomes an estimate).
#   Avatar-render row needs the `[render] first-frame` log line (musetalk_video.py) -> RESTART the avatar+
#   pipeline to pick it up. Missing/pre-t0/negative anchors render `unknown` (never a fake latency).
#   `pipeline/metrics.py` (the TtfoMeter) is deliberately UNTOUCHED -- the waterfall is derived in scripts/measure/.
#   (the two tools below are lower-level; run them standalone only for one-off debugging)
python -m scripts._webrtc_probe --mic output/q_ai.wav --lead 8     # drives a turn, records + metrics
E:\miniconda3\envs\musetalk\python.exe -m local_services.musetalk_server._capture output/q_ai.wav  # offline mp4
# A/V-SYNCED offline capture (keeps ONLY real video_start..video_end frames, auto-detects frame size):
E:\miniconda3\envs\musetalk\python.exe scripts\_capture_synced.py output/q_ai.wav

# FRAMES-vs-AUDIO + DRIFT method (how the P16 numbers were measured; no CosyVoice/pipeline/WebRTC):
# 1) start the MuseTalk server ALONE with the prod env (MUSETALK_TRT/FPS/SIZE from .env), MUSETALK_PROFILE=1
#    for per-8-frame-segment cost (logs feat/whisper/gpu/composite ms -> is render >= the fps budget?).
# 2) drive a WAV and read the THREE distinct counts + effective render fps:
python -m scripts._drive_frames output/reply_concise.wav 12          # paced (default) | burst (pure render)
#    - REAL rendered (server video_clock) = audio_sec*fps (+/-1)  <- lips are never short (P9/P10)
#    - DELIVERED > that by the pump's HELD/duplicate frames (frozen frame kept when render < fps)
#    - drift ~= audio_len * (1 - render_fps/fps); it only SCALES with turn length once render < fps
# 3) reproduce the shared-GPU drift offline (CosyVoice stand-in) — prove MUSETALK_TRT=1 holds >=fps:
E:\miniconda3\envs\musetalk\python.exe scripts\_gpu_contention_hog.py 4096   # run alongside step 2
# GOTCHA: _drive_frames paces the feed with ABSOLUTE deadlines, NOT cumulative asyncio.sleep(0.02) —
# on Windows the ~15ms timer granularity makes a cumulative-sleep feed ~40% slow and FAKES drift.

# Remote-link isolation test (streams a rendered mp4 LIVE as MJPEG, no GPU/WebRTC) — isolate link vs render:
python -m scripts.stream_live
```

`archive/` holds the regression tests kept out of the live tree: `_screech_repro_test.py`
(re-proves the steady-mode screech fix) and `_sync_routing_test.py`.

## Architecture — how one turn flows

`pipeline/main.py` assembles a linear Pipecat `Pipeline`; frames stream through it:

```
mic → transport.input()(+Silero VAD) → STT → aggregator.user()
    → LLM (streamed, sentence-aggregated) → TTS → Avatar → TtfoMeter
    → transport.output() → browser ;  aggregator.assistant() records the bot turn
```

Each stage is built by a thin factory in `pipeline/stages/` from `config` (one
provider, no branching). The whole thing streams: the LLM's first sentence
reaches TTS before the full answer exists, and TTS's first audio chunk reaches
the avatar immediately. `TtfoMeter` (`pipeline/metrics.py`) measures the gap from
`UserStoppedSpeakingFrame` → `BotStartedSpeakingFrame` (the <3 s metric).

**The avatar is a separate GPU process.** `local_services/musetalk_video.py`
(`MuseTalkVideoService`, the pipeline FrameProcessor) ↔ `local_services/musetalk_server/app.py`
(FastAPI ws server, `musetalk` env). Mouth-region lip-sync, portrait via
`AVATAR_REF`, port `:8002`. A load-time warmup renders 2 dummy segments through the FINAL
render path — it runs AFTER TRT init + free-torch + GPU-composite (2026-07-15; it used to run
before them, warming the torch UNet/VAE that `MUSETALK_FREE_TORCH` was about to free while
leaving the TRT engines' first execution cold for the first real turn). The wire contract:

- Client → server: a `config` json (incl. `"proto": 2` — see below), `speech_start`/`speech_end`/`reset` json,
  and binary **16 kHz mono PCM** chunks (the TTS audio, resampled client-side).
- Server → client: binary RGB frame buffers at a steady fps, plus
  `video_start`/`video_clock{frames}`/`video_end` markers (counting only *real* rendered frames).
- **proto 2 (2026-07-15, P51 — the live default between pipeline and server):** when the client's config asks
  `"proto": 2` and the server acks `{"type":"proto","v":2}`, every binary frame is prefixed with a 16-byte
  header (`MTF2` | kind u8 | audio_pos u64): kind 0 = real render / 1 = held re-send / 2 = idle-neutral, and
  `audio_pos` = cumulative REAL 16k samples of the turn covered once that frame shows. The steady client then
  releases voice paired to `audio_pos` (the server's own account of what it rendered) instead of `i/fps` index
  arithmetic, and held frames are declared instead of byte-compare-guessed (P39). An fps mismatch can no longer
  shift the audio↔lip mapping. Clients that never ask (offline harnesses `_capture.py`/`_drive_frames.py`/
  `_capture_synced.py`) keep the bare-frame wire byte-identical. Header carries a *position*, not audio bytes —
  the delivered voice stays the original 24 kHz TTS audio (bytes-in-packet would downgrade it to the 16 kHz
  lip-sync copy).

**A/V sync default = `steady`** (video-master): the voice is buffered and released **paced to the
real frames the server reports rendering**, so the voice waits when the render stalls and never
drifts ahead, for a synced start (the user's pick). `live` (audio-master) forwards the voice
immediately (lips best-effort, ~0.75s trail) and is the robust alternative that never pauses. The
client feeds audio to the server REAL-TIME-PACED (`_feed_q`), except the first `MUSETALK_FEED_BURST_S`
(1.0s) of each turn is burst un-paced so the renderer isn't starved at turn start (cut lip-start lag
~1.9s→~0.8s).

**CRITICAL COUPLING (`main.py`):** the per-frame A/V pinning (`sync_with_audio`) is a *no-op* unless
the transport is **non-live** — pipecat 1.3.0 only reads `_video_images` (where tagged frames land)
when `video_out_is_live=False`; with `is_live=True` the tagged frames are silently dropped and video
free-runs. So `video_out_is_live = not config.avatar_sync_with_audio` — never set `is_live`
independently. **One fps everywhere is load-bearing:** the server frame-drop stride, the client
release clock, and `main.py video_out_framerate` must all equal `config.avatar_fps` (MUSETALK_FPS) or
audio/video drift. (**P47.3:** the server's pump used to compute its frame interval ONCE, from its own env fps, *before* the
client's `{"type":"config","fps":N}` arrived — so it RENDERED at the client's fps and EMITTED at its own, drifting silently
whenever the two disagreed (e.g. the config panel writes `MUSETALK_FPS` to `.env` and restarts only the pipeline, leaving
`:8002` on the old value). It now recomputes `1/engine.fps` every tick, so the `config` message is fully honored. Still keep
the values equal — this closed the silent-failure mode, it did not make a mismatch *correct*.)

## Environment constraints / gotchas (READ before debugging the avatar)

- **PCM sample alignment is enforced at the PRODUCER — keep it there** (`cosyvoice_tts.py::run_tts`,
  `docs/PROBLEMS-AND-FIXES.md` **P40/P52**). Audio is int16 (2 bytes/sample), the server writes whole samples, but
  HTTP-chunk/TCP boundaries land mid-sample and aiohttp's `iter_chunked()` propagates them (measured live: several
  odd-length chunks per utterance, mid-stream). `run_tts` carries the dangling byte across reads so every
  `TTSAudioRawFrame` it yields is whole-sample — the ONE place the invariant is restored (2026-07-15; the old
  consumer-side patches `_align_even`/`_srv_carry` are removed, live-eye verified). If odd buffers ever reappear, fix
  the producer, never drop a byte downstream: a dropped byte assembles every following int16 from the wrong two bytes
  → the avatar server gets **loud broadband noise**, and since MuseTalk lip-syncs off a **Whisper of the waveform**,
  the mouth flaps in a generic wordless pattern that never closes for pauses (the voice still sounds perfect — that's
  the trap). This was THE multi-session "live lipsync is bad" bug.
- **Debugging the avatar's mouth: your reference must not share the suspect input** (P40 metrology). Three sessions were
  lost to tests that could not fail. "Delivered frames == offline render, byte-identical" only proves the render is
  **deterministic** — feed both sides the same corrupt PCM and it passes. An offline render fed a voice captured off
  **WebRTC** is fed the *repaired* downstream copy, so it bypasses the broken path and always looks good. Also: mouth-motion
  vs audio-RMS correlation is **useless** (it has misled 4×), and never verify A/V sync from a WebRTC capture reconstructed
  by *arrival* time (under `steady` the voice is released in bursts paced to the render). Use `MUSETALK_DUMP_PCM` +
  `MUSETALK_DUMP_DELIVERED` (uncompressed, what the browser actually gets). **Dead theories — do not re-open without new
  evidence:** fps/OOD-Whisper-stride (`MUSETALK_SIZE 512→256`), held-frame stalls, shared-GPU contention, VP8/transport,
  segmentation/Whisper context.
- **MuseTalk: `cudnn.benchmark` MUST stay `False`** (`musetalk_server/app.py`). With it `True`,
  cuDNN re-autotunes on the turn-START segment (different shape than mid-turn) → a **~16s GPU spike
  on the FIRST segment of every turn** → lips start ~5s late + the render falls behind on long
  replies ("audio ends, avatar keeps moving"). `False` removed it (steady-state per-frame time was
  unchanged). See `docs/PROBLEMS-AND-FIXES.md` P1. Diagnose render-stage timing with `MUSETALK_PROFILE=1`.
- **Judging audio garble: use a CONCATENATED WAV, never per-chunk RMS** — chunks aren't
  sample-aligned, so a single chunk reads as "loud garbage" even when the stream is clean (this
  cost hours; see PROBLEMS-AND-FIXES.md P3 method note).
- **onnxruntime / torch CUDA DLLs:** the avatar server adds torch's `lib/` dir to the DLL search
  path before importing onnxruntime, or onnxruntime silently falls back to CPU (~5× too slow,
  laggy/desynced avatar). Keep that.
- **`conda run` buffers stdout** — a running server's log looks empty; use the `-u` env-python
  invocation above for live logs.
- **NEVER do blocking I/O on the pipeline's event loop** (`docs/PROBLEMS-AND-FIXES.md` **P47.2**). One asyncio loop carries
  uvicorn's handlers, aiortc's RTP send/receive, the pipecat pipeline AND the MuseTalk websocket pump. A synchronous
  `urllib`/`requests` call inside an `async def` handler (or inside anything an async handler constructs) does not just stall
  that request — it **stops the loop**: no packets out, no audio pumped, the live call's voice and avatar freeze for the whole
  timeout. This is exactly what `_cloudflare_turn`'s `urlopen(timeout=8)` did from the `/client/ice-config` handler. Use
  `aiohttp` (already a dep) or `run_in_executor`, and cache failures so a dead endpoint can't be re-probed per request.
- **loguru is BRACE-style, not `%`-style** (**P47.4**). `logger.warning("x=%s", v)` formats via `str.format()`, so it prints the
  literal `%s` and silently DISCARDS `v`. A diagnostic written that way is worse than none — it looks like it fired and tells
  you nothing. Use f-strings (the house style everywhere else here).
- **The avatar server is single-client.** Fully close the browser tab between tries; a watchdog
  logs throughput and surfaces silent worker-thread crashes.
- **Windows console is cp1252** — `main.py` reconfigures stdout to UTF-8 so the Pipecat banner
  doesn't crash startup. Keep `.py` server source ASCII-safe.
- **conda env cert store:** the `musetalk`/`tts` conda envs have a broken Windows cert store (ssl
  ASN1) that kills torch.hub/urllib downloads — fix = curl-cache the weights + set `SSL_CERT_FILE`
  to certifi. See the `project-visualllm-conda-ssl-weights` memory.
- The `/client` UI is the **pipecat prebuilt bundle**, served as-is — don't add UI hacks back. The
  **one** sanctioned mechanism is the `<head>` script injection in `main.py`: env-gated patches
  register into the shared `_client_head_patches` list and ONE middleware serves the index with all
  of them (two separate index-serving middlewares would shadow each other). **As of 2026-07-14 only
  the TURN/ICE patch remains.** The other six installers (jitter buffer, phone-speaker route,
  video-stall monitor, A/V-stats monitor, playout probe, measure button) were REMOVED: they patched
  `/client` ONLY, and `MUSETALK_SPLIT=1` makes `/client` unsupported, so on the page actually used
  (`/studio`) they were inert -- `CLIENT_FORCE_SPEAKER=1` was loading nothing. The two that
  are real features (**jitter buffer** + **phone loudspeaker**) now live IN the static client, fed by
  `GET /client/ice-config`, which also serves `jitterBufferMs` + `forceSpeaker` alongside `iceServers`.
  The middleware itself STAYS -- it serves `/client/transcript`, `/say`, `/ice-config`,
  `/avatar-overlay` for studio. The index is served `Cache-Control: no-store`.
- **Open the client at `/client/` WITH the trailing slash** — the prebuilt page references its
  assets relatively, so `/client` (no slash) 404s them → white screen.
- **The custom client + AVATAR PRESETS live at `/studio/`** (`local_services/studio_client/`, mounted by
  `_install_studio_client`). Originally built as `/nimbus/` (figma-to-code redesign, `docs/PROBLEMS-AND-FIXES.md`
  P36/**P37**) for the female weather preset, then generalized 2026-07-11 to a white-theme copy for the
  **"Leo"** avatar (a man's face + cloned voice, built from a source clip). The two pages were ~740-line
  verbatim JS copies differing only in theme/branding, which is why `/nimbus/` was **removed 2026-07-14**
  — `/studio/` is now the single custom client for every avatar preset. It's a self-contained vanilla-JS
  page (no build step) that speaks the SAME SmallWebRTC signaling (`POST /api/offer`) as the prebuilt
  bundle, so it stays **additive**: `/client/` prebuilt is untouched and stays the fallback (though
  unsupported under `MUSETALK_SPLIT=1`, see above). Its extras are two thin server endpoints (same
  `_inject_client_patches` middleware pattern): **`POST /client/say {text}`** injects a typed turn via
  `LLMMessagesAppendFrame` into `_active_task`, and **`GET /client/transcript?since=N`** serves the
  conversation for the chat bubbles, fed by a READ-ONLY `BaseObserver` on the `PipelineTask`
  (`_TranscriptStore`; taps bot `LLMTextFrame`s + user `TranscriptionFrame`s — no pipeline structural
  change). Open it WITH the trailing slash, same as `/client/`.
  **Chat behavior (P37):** the user's speech streams into a LIVE bubble word-by-word (STT interims → the store's
  `_partial` slot → `/client/transcript` returns `"partial"` → the client polls at 200ms), then commits as **ONE**
  bubble per turn (segments accumulate, commit at `LLMFullResponseStart`; committing per `TranscriptionFrame` gives a
  bubble per speech pause). The user commit keys on the frame TYPE, **NOT** `frame.finalized` (Deepgram's streaming
  path leaves it False → gating on it dropped every user bubble). The mic button **mutes** (toggles the audio track,
  not disconnect) once connected. **Single-connection:** a new `/api/offer` disconnects the previous session
  (`_active_connection`) so two clients never fight the single-client avatar server.
  **No-mic fallback (2026-07-15):** if `getUserMedia` fails (classic case: an **RDP session without
  audio-recording redirection** — the box then has only a "Remote Audio" playback endpoint, so Chrome throws
  `NotFoundError: Requested device not found`), the client no longer refuses to connect. It warns "No microphone
  found — connecting anyway", sends `audio` as **recvonly** (the bot's voice + avatar video are inbound tracks,
  so nothing else changes), shows **"Type to talk"** instead of "Listening", and the chat box (`/client/say`)
  remains the input. Verified against the live pipeline with the mic-less `_webrtc_probe` (both tracks arrive,
  greeting audio plays). To get a real mic over RDP: mstsc → Local Resources → Remote audio → Settings →
  "Record from this computer".
  **Two rules the transcript path has now paid for in blood — do not undo either:**
  - **Dedupe frames on `frame.id`, NEVER `id(frame)`** (`docs/PROBLEMS-AND-FIXES.md` **P45**). The observer must dedupe
    (one frame OBJECT is pushed by several processors, so it is seen more than once), but `id(frame)` is the MEMORY
    ADDRESS — CPython recycles a freed frame's address into the next frame, so new tokens hit the "already seen" set and
    were **silently deleted from the bubble** (`自然語言處理` → `自言處理`). Pipecat stamps every frame with a monotonic
    unique `frame.id`; use it. Reset `_seen` at `LLMFullResponseEndFrame` (ids are monotonic, so it can only grow).
    **The voice is unaffected by this class of bug** (TTS reads the frames itself) — a corrupt bubble looks like a bad LLM,
    which is exactly why it hid.
  - **The transcript poll must be re-entrant-safe AND idempotent** (**P46**). `setInterval(poll, 200)` fires whether or not
    the last poll returned, and `transcriptSeq` only advances once a response lands — so one >200 ms browser main-thread
    hitch sends the next poll out with the STALE `since`, the server returns the same rows to both, and both render them
    (**identical duplicate bubbles**). Keep the `polling` in-flight flag AND the `it.seq > transcriptSeq` skip.
  **Not a bug, but it looks like one:** on SPEAKERS the live mic transcribes the avatar's own voice (`ECHO_GUARD=0`
  baseline; `=1` is mechanically fixed since P53 but not yet ear-tested) → junk "user" turns (`奶奶有皮革戀愛`) → a
  junk query bubble + a real extra reply, and under
  `ALLOW_INTERRUPTIONS=1` it truncates the bot mid-sentence. Use headphones. Duplicate bubbles with IDENTICAL text = the
  poll race (P46); a gibberish second query bubble = mic echo.
- **AVATAR PRESETS (2026-07-11).** Because one GPU runs one avatar, `/studio/` shows whichever **preset**
  is live. A preset = a full backend swap (portrait `AVATAR_REF` + cloned voice
  `COSYVOICE_PROMPT_WAV/TEXT` + `LANGUAGE`), defined in `config_panel/server.py::PRESETS`
  (`nimbus` = female weather en; `leo` = his face + voice, zh — both open at `/studio/`). The config
  panel's **"Avatar preset"** card
  (`POST /preset`) writes those `.env` keys + `AVATAR_PRESET`, writes the CJK transcript to a WSL-sourced file
  (`.preset_voice.env` — keeps CJK OFF the mangling-prone WSL command line), then relaunches **CosyVoice → avatar →
  pipeline in the P15 load order** (frees `:8002` VRAM first). Leo's portrait is the full rectangular source frame (the
  server derives the split bbox+background from it). His voice reference must be **Simplified** zh (the clean CosyVoice
  path; Traditional garbles, P43) with an accurate transcript (verified by transcribe-back). ~2–4 min to switch.
- **Fullscreen-avatar button (2026-07-11)** — `/studio/` has a top-right button that fullscreens the
  **`.presenter`** (bg + mouth-crop + name tag together, not the bare `<video>`). On `fullscreenchange` it recomputes
  `layoutSplitVideo()` so the split mouth-crop re-scales/re-positions onto the face at the new size (else the mouth lands
  wrong fullscreen). Handles standard + `webkit` fullscreen APIs; static files, so a reload picks it up (no restart).

## Conventions

- Keep stage factories single-provider and thin; config is `.env`-driven only.
- Comments state the *why* (latency, a Pipecat quirk, a hardware constraint) — match that voice.
- Accepted tradeoffs (see `STATUS.md`): echo-guard defaults OFF (`ECHO_GUARD=0`, barge-in — use
  headphones; the P44 baseline). The old steady-sync blocker for `=1` (P11 stuck-mute — no BotStopped
  ever fired) is **FIXED at the root (P53, 2026-07-15)**: the avatar client holds the per-turn
  `TTSStoppedFrame` until the voice fully drains, so BotStoppedSpeaking fires at true end of speech
  under steady (probe + log verified; `archive/_tts_stop_order_test.py` is the regression proof).
  `=1` under steady is now mechanically sound but **not yet live-ear-verified** — test a real
  echo-guard session before relying on it. **`ALLOW_INTERRUPTIONS` (default `1` = BASELINE since P44, was live `.env`
  `0` under P37):** `0` = the bot always finishes its reply (user speech during playback never cancels
  it; typed turns queue politely). `1` = a mid-reply interruption **flushes the current turn clean**.
  This flips the turn-START strategies' `enable_interruptions` (the barge-in broadcast) — NOT the
  echo-guard mic mute (P11, root-cause-fixed by P53), so it is safe under steady. **Two interrupt fixes shipped under it (P44):** (a) a
  clean-FLUSH — on the pipecat `InterruptionFrame` the avatar client drops in-flight server frames
  (`_flushing`, `musetalk_video.py`) and the MuseTalk server drains its `out_q` on `reset`
  (`app.py`, reuses the `seg_restart` path), so the avatar no longer keeps lip-moving silently / leaks
  the old turn into the next; (b) a TYPED turn now barges in too — `/client/say` (the `/studio` keyboard
  path) emits an `InterruptionFrame` before the `LLMMessagesAppendFrame` when `allow_interruptions`
  (a typed turn otherwise emits no barge-in, so the old turn played to completion). The partial bot
  reply is preserved in context automatically (pipecat's assistant aggregator commits it on
  interruption). On the single shared GPU the
  lips can trail the voice under load in `live` mode — that's the cost of `live` never freezing; the
  SAFE next lever is bounding the avatar server's `out_q`, **never** re-locking the voice (locked
  sync froze it — see STATUS.md).
- Pipecat import paths drift between releases; the fragile ones are isolated to
  `pipeline/stages/*.py`, `pipeline/main.py`, `pipeline/metrics.py`. Run `python -m scripts.preflight`
  after touching them.
- **Remote viewing** (RDP into this box, or the live avatar over a `tailscale serve` HTTPS URL in a
  remote browser) has its own pitfalls: RDP adds video choppiness AND
  desyncs audio/video — when judging avatar smoothness/sync, use `_capture.py` (offline, no
  WebRTC/RDP) or a native remote browser, never the RDP window; re-encode any mp4 trims
  (`-ss -c copy` breaks playback). When the avatar "won't show" or "won't talk," first check both
  processes are up (`:7860` and `:8002`) and that the pipeline picked up the latest code (restart
  it; a stale process lacks recent fixes).
