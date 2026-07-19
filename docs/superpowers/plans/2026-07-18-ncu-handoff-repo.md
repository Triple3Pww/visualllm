# NCU Handoff Repo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline, recommended here — the cleanup and doc-writing need the in-context project knowledge; cold subagents would re-derive it expensively). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Produce `E:\Claude\VisualLLm-NCU`, a clean, curated, self-contained copy of the working avatar system with cleaned code and outside-reader docs, ready to hand to NCU.

**Architecture:** Copy the in-scope source trees from `E:\Claude\VisualLLm` into a fresh sibling repo, delete the out-of-scope subsystems, surgically clean the carried-over code of orphaned references and dead knobs, replace the session-log docs with five outside-reader docs, gitignore the heavy assets, and verify by preflight + a driven turn.

**Tech Stack:** Python 3.11 (pipeline, system Python), conda envs (musetalk / tts), Pipecat 1.3.0, WSL2 + vLLM (CosyVoice), MuseTalk, sherpa-onnx, PowerShell tooling. No build/lint/test suite exists — verification is `scripts.preflight` + behavior.

## Global Constraints

- Source repo `E:\Claude\VisualLLm` stays **untouched** — it is the reference. All edits happen in `E:\Claude\VisualLLm-NCU`.
- Cleanup is **surgical**: remove orphans from the scope reduction + genuine dead code only. Do NOT rewrite working logic or re-format untouched code.
- **Preserve load-bearing "why" comments** (latency, Pipecat quirks, hardware constraints). Keep `MUSETALK_DUMP_*` diagnostics (P40 mouth-debug).
- Windows: write `.py` server source ASCII-safe; `.env`/scripts UTF-8 without BOM; `.ps1` ASCII-only.
- Fresh `git init`, single initial commit at Task 1, then a commit per task.
- Heavy assets (upstream `CosyVoice/`, MuseTalk vendor + weights, `models/`, `trt_cache/`) are gitignored and documented-download — never copied into git.
- Spec: `docs/superpowers/specs/2026-07-18-ncu-handoff-repo-design.md`.

---

### Task 1: Scaffold the new repo and copy in-scope source

**Files:**
- Create dir: `E:\Claude\VisualLLm-NCU\`
- Copy (source-only) the in-scope trees listed below.

**Steps:**

- [ ] **Step 1: Create the target dir and copy in-scope trees.** From a Git-Bash shell, copy each tree, excluding heavy/removed content. Use `rsync`-style excludes via `cp` + follow-up deletes (Task 2 prunes the rest):

```bash
SRC="/e/Claude/VisualLLm"; DST="/e/Claude/VisualLLm-NCU"
mkdir -p "$DST"
# top-level source files
cp "$SRC"/requirements.txt "$SRC"/log_setup.py "$DST"/
cp "$SRC"/"Run VisualLLm.exe" "$DST"/ 2>/dev/null || true
# pipeline (source only; __pycache__ excluded after)
cp -r "$SRC"/pipeline "$DST"/pipeline
cp -r "$SRC"/local_services "$DST"/local_services
cp -r "$SRC"/scripts "$DST"/scripts
cp -r "$SRC"/assets "$DST"/assets
cp -r "$SRC"/tts "$DST"/tts
```

- [ ] **Step 2: Strip pycache, logs, and heavy caches immediately** so nothing heavy is staged:

```bash
cd "$DST"
find . -type d -name __pycache__ -prune -exec rm -rf {} +
find . -type d -name '.pytest_cache' -prune -exec rm -rf {} +
rm -rf tts/cosyvoice-server/CosyVoice tts/cosyvoice-server/outputs tts/cosyvoice-server/output tts/cosyvoice-server/logs
rm -rf local_services/musetalk_server/vendor local_services/musetalk_server/trt_cache local_services/musetalk_server/avatar_cache
rm -f local_services/musetalk_server/*.log
```

- [ ] **Step 3: Verify the copy landed and is light.**

Run: `cd "$DST" && du -sh . && ls`
Expected: total well under ~500 MB (no CosyVoice/vendor/trt_cache); top-level shows `pipeline local_services scripts assets tts requirements.txt log_setup.py`.

- [ ] **Step 4: git init + baseline .gitignore** (the real `.gitignore` is finalized in Task 5; a minimal one now keeps the first commit clean):

```bash
cd "$DST"
printf '%s\n' '__pycache__/' '*.pyc' '.env' '*.log' 'logs/' 'models/' 'output/' 'state/' \
  'local_services/musetalk_server/vendor/' 'local_services/musetalk_server/trt_cache/' \
  'local_services/musetalk_server/avatar_cache/' 'tts/cosyvoice-server/CosyVoice/' > .gitignore
git init -q && git add -A && git commit -qm "chore: initial import of in-scope source from VisualLLm" && echo COMMITTED
```

Expected: `COMMITTED`, and `git ls-files | wc -l` is a few hundred (source files), not thousands.

---

### Task 2: Prune out-of-scope subsystems and dev cruft

**Files (delete in `VisualLLm-NCU`):**
- `local_services/avatar_memory.py`
- `local_services/jaitts_server/`
- `local_services/nimbus_client/`
- `tts/cosyvoice-server/` dev probes: `_*.py`, `_*.sh`, `test_*.py`, `benchmark.py`, `test_engine_dispatch.py`
- `local_services/musetalk_server/`: `_capture.py` stays (offline capture is a documented tool); delete `*.err.log`/`*.out.log`/`smoke.log` if any remain
- Any copied `paper/ archive/ learn/ research/ scratchpad_tts/ tools/ state/ output/` (should not have been copied, but confirm)

**Steps:**

- [ ] **Step 1: Delete the removed subsystems.**

```bash
cd /e/Claude/VisualLLm-NCU
rm -f  local_services/avatar_memory.py
rm -rf local_services/jaitts_server local_services/nimbus_client
rm -f  tts/cosyvoice-server/_*.py tts/cosyvoice-server/_*.sh tts/cosyvoice-server/test_*.py tts/cosyvoice-server/benchmark.py
rm -rf paper archive learn research scratchpad_tts tools state output   # no-op if absent
```

- [ ] **Step 2: Verify none of the removed trees remain.**

Run: `ls local_services/ | grep -E 'avatar_memory|jaitts|nimbus'` (expect no output) and `ls tts/cosyvoice-server/*.py` (expect only `app.py tts_engine.py client_example.py`).

- [ ] **Step 3: Commit.**

```bash
git add -A && git commit -qm "chore: remove out-of-scope subsystems (growing-memory, Thai JaiTTS, nimbus client, dev probes)" && echo COMMITTED
```

---

### Task 3: Clean `pipeline/` code

**Files:**
- Modify: `pipeline/main.py`, `pipeline/config.py`, `pipeline/stages/stt.py`, `pipeline/stages/tts.py`, `pipeline/stages/llm.py`, `pipeline/stages/vad.py`, `pipeline/stages/avatar.py`, `pipeline/metrics.py`

**Cleanup rules (apply by reading each file):**
1. Remove the `_install_nimbus_client` call + function and any `nimbus_client` references in `main.py`.
2. Remove `avatar_memory` / `MemoryStore` imports and wiring in `main.py` and `config.py` (the growing-memory harness). `weather_chain` LLM wiring **stays** (it is the kept example).
3. Remove the `jaitts` branch from `stages/tts.py` (keep `cosyvoice`; a bad `TTS_PROVIDER` still raises).
4. Remove `.env` config fields in `config.py` that have no remaining reader after 1–3 (e.g. memory knobs, jaitts knobs). Grep each field name across the repo before deleting.
5. Fix/drop comments that reference removed code; keep latency/Pipecat/hardware "why" comments verbatim.

**Steps:**

- [ ] **Step 1:** Read `pipeline/main.py` fully; apply rules 1–2 and 5. Read `pipeline/config.py`; apply rules 2, 4, 5. Read `pipeline/stages/tts.py`; apply rule 3. Read `stt.py`/`llm.py`/`vad.py`/`avatar.py`/`metrics.py` and apply rule 5 only (they should need little).

- [ ] **Step 2: Grep-verify no orphan references remain in `pipeline/`.**

Run: `grep -rniE 'avatar_memory|memorystore|jaitts|nimbus' pipeline/`
Expected: **no output**.

- [ ] **Step 3: Import-check the pipeline modules** (system Python 3.11):

Run: `python -c "import ast,glob; [ast.parse(open(f,encoding='utf-8').read()) for f in glob.glob('pipeline/**/*.py',recursive=True)]; print('AST OK')"`
Expected: `AST OK` (syntax valid after edits).

- [ ] **Step 4: Commit.**

```bash
git add -A && git commit -qm "refactor(pipeline): remove memory/jaitts/nimbus orphans and dead knobs" && echo COMMITTED
```

---

### Task 4: Clean `local_services/` code

**Files:**
- Modify: `local_services/config_panel/server.py`, `local_services/config_panel/index.html`, `local_services/studio_client/index.html`, `local_services/musetalk_video.py`, `local_services/musetalk_server/app.py`, `local_services/cosyvoice_tts.py`, `local_services/sherpa_stt.py`, `local_services/first_piece_aggregator.py`, `local_services/weather_chain_llm.py` (keep — light touch), `local_services/README.md`

**Cleanup rules:**
1. `config_panel/server.py`: remove the growing-memory card/knobs, the Thai/jaitts references, and any `nimbus` mentions. **Keep the avatar-preset card** (nimbus/leo presets stay). Remove the `memory-sim :7900` health dot if still present.
2. `config_panel/index.html` + `studio_client/index.html`: remove UI referencing removed subsystems (memory, Thai); keep preset UI, mouth-crop, fullscreen, transcript, no-mic fallback.
3. `musetalk_video.py`, `musetalk_server/app.py`: no subsystem refs expected — apply comment rule 5 only; **do not touch** the proto-2 / sync / P-guard logic.
4. `cosyvoice_tts.py`: keep the producer-side odd-byte carry (P52) and its "why" comment verbatim.
5. `local_services/README.md`: rewrite to describe only the kept services.

**Steps:**

- [ ] **Step 1:** Read and clean `config_panel/server.py` (rule 1), then its `index.html` (rule 2), then `studio_client/index.html` (rule 2).

- [ ] **Step 2:** Read `musetalk_video.py`, `musetalk_server/app.py`, `cosyvoice_tts.py`, `sherpa_stt.py`, `first_piece_aggregator.py`, `weather_chain_llm.py`; apply comment rule 5 only, preserving all P-guard logic + why-comments.

- [ ] **Step 3:** Rewrite `local_services/README.md` for the kept services only.

- [ ] **Step 4: Grep-verify.**

Run: `grep -rniE 'avatar_memory|memorystore|jaitts|nimbus|memory-sim|:7900' local_services/`
Expected: no output (a benign match inside a preserved unrelated word is fine — inspect if any).

- [ ] **Step 5: AST-check + commit.**

```bash
python -c "import ast,glob; [ast.parse(open(f,encoding='utf-8').read()) for f in glob.glob('local_services/**/*.py',recursive=True)]; print('AST OK')"
git add -A && git commit -qm "refactor(local_services): drop memory/jaitts/nimbus refs; keep presets, mouth-crop, P-guards" && echo COMMITTED
```

---

### Task 5: Clean scripts, TTS wrapper, and repo config; re-point paths

**Files:**
- Modify: `scripts/launch.ps1`, `scripts/run.ps1`, `scripts/tunnel.ps1`, `scripts/preflight.py`, `scripts/measure/*.py`, `scripts/*.py` probes
- Modify: `tts/cosyvoice-server/run_vllm_server.sh`, `tts/cosyvoice-server/app.py`, `tts/cosyvoice-server/README.md`
- Create/replace: `.gitignore`, `.env.example`, `requirements.txt` (verify)

**Cleanup rules:**
1. **Re-point absolute paths** from `/mnt/e/Claude/VisualLLm/...` and `E:\Claude\VisualLLm\...` to the new repo root in `run_vllm_server.sh`, `launch.ps1`, `config_panel/server.py` (already touched), and any doc string. Prefer `SCRIPT_DIR`-relative where the file already supports it.
2. Remove references to removed subsystems / `jaitts` server start in `launch.ps1`/`run.ps1`.
3. `scripts/measure/`: keep as-is (in scope) — comment rule 5 only.
4. Write a full **`.gitignore`** = the source repo's plus nothing memory-specific removed (state/ stays ignored harmlessly).
5. Write a clean **`.env.example`** derived from the source repo's, dropping every key with no reader after Tasks 3–4 (memory, jaitts). Keep the CosyVoice / MuseTalk / preset / measure / interruption / TURN knobs.

**Steps:**

- [ ] **Step 1:** Read `run_vllm_server.sh` + `launch.ps1` + `run.ps1`; apply rules 1–2. Confirm WSL path token is the new repo (`/mnt/e/Claude/VisualLLm-NCU/tts/cosyvoice-server/`).

- [ ] **Step 2:** Copy the source `.gitignore` content into the new repo's `.gitignore` (supersede the Task-1 stub). Write `.env.example` cleaned to scope (read the source `.env.example`, drop dead keys, keep a one-line comment per kept key).

- [ ] **Step 3: Grep-verify no stale absolute path or removed-subsystem ref in scripts/tts.**

Run: `grep -rniE 'VisualLLm(\\\\|/)tts|/mnt/e/Claude/VisualLLm/|jaitts|avatar_memory' scripts/ tts/ | grep -v VisualLLm-NCU`
Expected: no output (every path points at `VisualLLm-NCU`).

- [ ] **Step 4: Commit.**

```bash
git add -A && git commit -qm "chore(scripts/tts/config): re-point paths to new repo, clean .env.example + .gitignore" && echo COMMITTED
```

---

### Task 6: Docs — README + ARCHITECTURE

**Files:**
- Create: `README.md`, `docs/ARCHITECTURE.md`

**Source material to distill from (read in the SOURCE repo):** `CLAUDE.md` (stack table, wire contract, sync), `WORKFLOW.md` (turn flow, §8 knobs), `docs/GPU-REQUIREMENTS.md`, `docs/LATENCY-MATRIX.md`.

**Steps:**

- [ ] **Step 1: Write `README.md`** — sections: one-paragraph what-it-is; the stage table (VAD/STT/LLM/TTS/Avatar with where each runs); an architecture-at-a-glance mermaid diagram of the turn flow; a repo map (one line per top dir); a "Start here → docs/SETUP.md" pointer; hardware one-liner (one 16 GB GPU, ~5.6 GB used; WSL2). No session-log voice — present tense, outside reader.

- [ ] **Step 2: Write `docs/ARCHITECTURE.md`** — sections: turn flow (the `mic → VAD → STT → aggregator → LLM → TTS → Avatar → TtfoMeter → transport` chain); the thin stage-factory pattern (one provider per stage, `.env`-selected); the avatar as a separate GPU process + why; the client↔server wire contract (proto 2 header: kind + audio_pos); A/V sync `steady` vs `live` and the `video_out_is_live = not sync_with_audio` coupling; the streaming/TTFO design (first sentence → TTS before full answer). Use the CLAUDE.md "Architecture — how one turn flows" section as the factual base; rewrite for an outside reader (no P-numbers in prose; move gotchas to ENGINEERING-NOTES).

- [ ] **Step 3: Verify** both files render (headings, mermaid fenced) — `grep -c '^#' README.md docs/ARCHITECTURE.md` > 0.

- [ ] **Step 4: Commit.** `git add -A && git commit -qm "docs: README + ARCHITECTURE for outside readers" && echo COMMITTED`

---

### Task 7: Docs — SETUP + USAGE

**Files:**
- Create: `docs/SETUP.md`, `docs/USAGE.md`

**Source material:** SOURCE `SETUP.md`, `docs/GPU-REQUIREMENTS.md`, `CLAUDE.md` "Commands" section, `WORKFLOW.md`, the memory `project-visualllm-conda-ssl-weights` (SSL/weights gotcha), `project-visualllm-cosyvoice-vllm` (WSL build).

**Steps:**

- [ ] **Step 1: Write `docs/SETUP.md`** — from-zero, ordered: (a) hardware requirements (GPU VRAM, disk for weights, WSL2); (b) the three Python environments — system Python 3.11 (pipeline), `musetalk` conda env, `tts`/WSL `cosyvllm` conda env — with create commands; (c) heavy-asset downloads with exact commands: upstream CosyVoice + weights, MuseTalk weights (`download_weights`), sherpa-onnx model into `models/`, the conda-env SSL cert fix (`SSL_CERT_FILE`=certifi + curl-cache s3fd/2DFAN4); (d) `.env` from `.env.example` + the WSL-IP-not-localhost rule; (e) run + `python -m scripts.preflight` verify; (f) troubleshooting (silent bot = `:8001` down; white screen = missing trailing slash; laggy avatar = onnxruntime on CPU).

- [ ] **Step 2: Write `docs/USAGE.md`** — the three processes (WSL CosyVoice, MuseTalk avatar server, pipeline) + the one-click launcher (`Run VisualLLm.exe` / `launch.ps1`); the config panel (`:7870`); `/studio` vs `/client` (trailing slash); the curated `.env` knob reference (LANGUAGE, TTS/STT/LLM providers, MUSETALK_SYNC_MODE, MUSETALK_SPLIT, MUSETALK_SIZE/FPS/LEAD_FRAMES, ALLOW_INTERRUPTIONS, the preset system, WEBRTC_PUBLIC/TURN) — table form, one "what it does" line each, grouped; the measure harness (`python -m scripts.measure --turns 5`).

- [ ] **Step 3: Commit.** `git add -A && git commit -qm "docs: SETUP + USAGE" && echo COMMITTED`

---

### Task 8: Docs — EXTENDING + ENGINEERING-NOTES

**Files:**
- Create: `docs/EXTENDING.md`, `docs/ENGINEERING-NOTES.md`

**Source material:** SOURCE `local_services/weather_chain_llm.py` (the example), `pipeline/stages/*.py` (factory pattern), `config_panel/server.py::PRESETS` (preset shape), `CLAUDE.md` gotchas, `docs/PROBLEMS-AND-FIXES-CLEAN.md`, the ENGINEERING-NOTES groupings in the spec §5.

**Steps:**

- [ ] **Step 1: Write `docs/EXTENDING.md`** — (a) the stage-factory contract: each `pipeline/stages/<stage>.py` is a thin factory returning a Pipecat service selected by `.env`; show the LLM slot's contract (`LLMFullResponseStartFrame → LLMTextFrame* → LLMFullResponseEndFrame` per `LLMContextFrame`); (b) **the worked example: connect your own LLM** — walk through `weather_chain_llm.py` (`WeatherChainLLMService`, streaming an external LangServe SSE endpoint, tolerant SSE parsing), then the two ways NCU plugs in their backend: OpenAI-compatible (just set `OPENROUTER_BASE_URL`) vs a custom `LLMService` subclass like weather_chain; (c) how to add an avatar preset (`PRESETS` entry: portrait `AVATAR_REF` + voice ref + language; Simplified-zh voice ref caveat).

- [ ] **Step 2: Write `docs/ENGINEERING-NOTES.md`** — distilled, grouped, outside-reader lessons (drop P-numbers, keep the mechanism + the fix):
  - *Audio integrity*: PCM must be whole-sample; the fix belongs at the producer (`cosyvoice_tts.run_tts` byte-carry); judge garble on a concatenated WAV, never per-chunk RMS; MuseTalk lip-syncs off a Whisper of the waveform, so corrupt PCM = flapping mouth with perfect-sounding voice (the trap).
  - *Avatar / GPU*: `cudnn.benchmark=False` (else a ~16 s first-segment spike); TensorRT buys render margin, not a visible fix; VRAM budget on one 16 GB card (vLLM util + free-torch); `MUSETALK_SIZE=512` is the lag-free ceiling on the shared GPU; `MUSETALK_FPS` must match everywhere.
  - *Networking / event loop*: CosyVoice reached via the WSL IP, never localhost (relay buffers ~2 s); never do blocking I/O on the pipeline event loop (freezes audio+video); TURN/ICE for public links.
  - *Latency methodology*: the probe passes what the eye rejects (verify with a live eye / delivered capture, not just probe logs); measure a fresh session (per-connection drift); the TTFO metric is blind before t0.
  - *Debugging the avatar mouth*: your reference must not share the suspect input; use `MUSETALK_DUMP_PCM`/`MUSETALK_DUMP_DELIVERED`.

- [ ] **Step 3: Commit.** `git add -A && git commit -qm "docs: EXTENDING (connect-your-own-LLM example) + ENGINEERING-NOTES" && echo COMMITTED`

---

### Task 9: Verify the clean repo behaves, then final commit

**Steps:**

- [ ] **Step 1: Whole-repo orphan grep (final gate).**

Run (from `VisualLLm-NCU`): `grep -rniE 'avatar_memory|memorystore|\bjaitts\b|nimbus_client' --include=*.py --include=*.html --include=*.ps1 --include=*.sh .`
Expected: **no output**. (A prose mention in a doc explaining what was removed is acceptable; code refs are not.)

- [ ] **Step 2: Preflight (the sanctioned Pipecat-drift check).** Run from `VisualLLm-NCU` with system Python 3.11:

Run: `python -m scripts.preflight`
Expected: it reports every fragile import resolving (no `ImportError`). If it fails on a removed-module import, fix the straggler and re-run.

- [ ] **Step 3: Drive one real turn** (requires the live stack — WSL CosyVoice `:8001`, MuseTalk `:8002`, pipeline). If the environment is up:

Run: `python -m scripts.measure --no-browser --turns 1`
Expected: a waterfall row with a non-null TTFO. If the stack cannot be started from here, record that explicitly in the hand-back message and rely on Steps 1–2 + the source repo's proven behavior.

- [ ] **Step 4: Final commit + summary.**

```bash
git add -A && git commit -qm "chore: verification pass (preflight clean, orphan-grep clean)" && echo COMMITTED
git log --oneline | cat
```

Expected: a short, clean commit history (~9 commits), no heavy files tracked (`git ls-files | wc -l` a few hundred).

---

## Self-Review

**Spec coverage:** scaffold+copy (T1), removals (T2), pipeline clean (T3), local_services clean (T4), scripts/tts/config clean + path re-point (T5), all five docs (T6–T8: README, ARCHITECTURE, SETUP, USAGE, EXTENDING, ENGINEERING-NOTES), heavy-asset gitignore (T1/T5), verification incl. driven turn (T9), fresh git + per-task commits (all). weather_chain-as-example (T8 §1b). Presets/mouth-crop/measure kept (T2/T4/T5). Every spec section maps to a task.

**Placeholder scan:** no TBD/TODO; every step has the concrete command or the concrete doc outline. Doc tasks specify exact sections + the source files to distill from (the executor is the same in-context agent, so section-level outlines are actionable, not vague).

**Consistency:** target path `E:\Claude\VisualLLm-NCU` / WSL `/e/Claude/VisualLLm-NCU` used throughout; grep patterns consistent (`avatar_memory|memorystore|jaitts|nimbus`); `weather_chain` consistently KEPT, `avatar_memory` consistently REMOVED.
