# VRAM Footprint Trim + Measured Lag Revisit — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reclaim the realistic ~2–3 GB on the shared 16 GB card via reversible knobs, each gated on a stop-and-diff VRAM measurement, and make a measured `live`-vs-`steady` call on the ~4 s avatar lag — without changing the steady default.

**Architecture:** Two GPU processes share one card — CosyVoice on vLLM (WSL) and MuseTalk (Windows `musetalk` env). We trim each side's *reserved/KV/activation* headroom (the model weights are a hard floor), measure every change by stopping pieces and diffing `nvidia-smi memory.used`, then re-measure the `live`-mode lip-trail after bounding the avatar `out_q`. All changes are env/arg knobs; nothing is hardcoded.

**Tech Stack:** vLLM `EngineArgs` (CosyVoice repo), PyTorch CUDA allocator (`empty_cache`), FastAPI/asyncio MuseTalk server, `nvidia-smi` (Windows + WSL), `scripts/measure.py` A/V harness.

## Global Constraints

- **No unit-test suite** — CLAUDE.md forbids inventing one. Each task's verification is a **measurement gate**: `nvidia-smi` stop-and-diff and/or a render/synthesis check. "Expected: PASS" means the measurement met the gate.
- **Measurement = stop-and-diff only.** Per-process `used_memory` reads `[N/A]` on this WDDM card (Windows **and** WSL). Attribute VRAM by stopping a piece and diffing `memory.used`. Query: `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`.
- **Noise floor:** desktop VRAM drifts ~±170 MiB. Treat any delta **< 200 MiB** as noise, not a win.
- **Load order is load-bearing:** start **CosyVoice (vLLM) BEFORE MuseTalk**, always. Clean recovery = stop all → start cosyvoice on the near-empty card → then `scripts/run.ps1`.
- **vLLM util safe floor ≈ 0.25.** `0.20` (3.26 GB) crashed ("No available memory for the cache blocks"). Never drop below a value where the vLLM startup log's "Available KV cache memory" line stays **positive**.
- **Steady stays the default.** `MUSETALK_SYNC_MODE=steady` is unchanged; `live` is only *offered* for the user to judge. **Never re-lock the voice to video** (it froze the voice — confirmed).
- **ASCII-only** in edited `.py`/`.sh` server source; UTF-8 **without BOM**.
- **Branch:** `feat/offline-stt-sensevoice` (continue here — the related `docs/gpu-memory-notes.md` edits already live on it).
- **Reversibility:** every knob keeps its current default; we change launch env / add an env-gated arg, so a revert is a one-line change.

**Launch commands (referenced throughout):**
- CosyVoice vLLM (WSL): `wsl -d Ubuntu -e bash -c "bash /mnt/e/Claude/cosyvoice-local-tts/run_vllm_server.sh"`
- MuseTalk (live logs): `E:\miniconda3\envs\musetalk\python.exe -u -m local_services.musetalk_server.app`
- Avatar server + pipeline together: `.\scripts\run.ps1`
- A/V timing harness: `python -m scripts.measure --offline-capture`

---

### Task 0: Capture the baseline (the gate everything compares against)

**Files:**
- Create: `docs/superpowers/vram-baseline.md` (scratch record of measured numbers; deleted/folded into gpu-memory-notes.md in Task 6)

**Interfaces:**
- Produces: a table of `memory.used` (MiB) at four states + the current `live` trail (seconds), referenced by Tasks 1–6 as "baseline".

- [ ] **Step 1: Ensure the GPU is free and the stack is down.** Close `/client` tabs. Stop pipeline, MuseTalk, and WSL vLLM. Confirm:

```
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
```
Expected: ~942 MiB (desktop only). Record as `S0_down`.

- [ ] **Step 2: Start CosyVoice vLLM only, measure.**

```
wsl -d Ubuntu -e bash -c "bash /mnt/e/Claude/cosyvoice-local-tts/run_vllm_server.sh"
```
Wait for `/health` (`:8001`). Then run the `nvidia-smi` query. Record as `S1_vllm`. Also copy the vLLM startup log line **"Available KV cache memory: … GiB"** — record the number.

- [ ] **Step 3: Start MuseTalk + pipeline, measure idle and mid-turn.**

```
.\scripts\run.ps1
```
After "MuseTalk ready", run the query → record `S2_idle`. Then drive one turn (`python -m scripts.measure --offline-capture`) and, while it renders, run the query again → record `S2_midturn` (peak).

- [ ] **Step 4: Measure the current `live`-mode trail.** Set `MUSETALK_SYNC_MODE=live` in `.env`, restart the pipeline, run `python -m scripts.measure --offline-capture`, and read the lip-start-vs-voice and steady-state trail from the report. Record `live_trail_baseline`. Revert `.env` to `steady`.

- [ ] **Step 5: Write the baseline table.**

```markdown
# VRAM baseline (measured YYYY-MM-DD)
| State | memory.used (MiB) |
|-------|-------------------|
| S0_down (desktop only) | <n> |
| S1_vllm (vLLM only) | <n> |
| S2_idle (+MuseTalk+pipeline) | <n> |
| S2_midturn (peak) | <n> |
vLLM "Available KV cache memory": <n> GiB
live_trail_baseline: lips start +<n>s, steady-state trail ~<n>s
```

- [ ] **Step 6: Commit.**

```bash
git add docs/superpowers/vram-baseline.md
git commit -m "chore(vram): record measured GPU baseline before trim"
```

---

### Task 1: vLLM `gpu_memory_utilization` step-down (Workstream A1)

**Files:**
- Modify: `E:\Claude\cosyvoice-local-tts\run_vllm_server.sh` (add an explicit `export COSYVOICE_VLLM_GPU_UTIL`)

**Interfaces:**
- Consumes: `S1_vllm` baseline + the baseline KV-cache GiB from Task 0.
- Produces: the chosen util value + measured `S1_vllm` at that value (`S1_vllm@util`), used by Task 6.

- [ ] **Step 1: Add the explicit export to the launch script.** In `run_vllm_server.sh`, after the existing `export COSYVOICE_VLLM=1` block, add:

```bash
# VRAM trim: cap the fraction of the 16GB card vLLM may use. 0.30 is the prior
# default (~4.89GB); step down toward the ~0.25 safe floor while the startup log's
# "Available KV cache memory" stays positive. Read by CosyVoice/cosyvoice/cli/model.py.
export COSYVOICE_VLLM_GPU_UTIL=${COSYVOICE_VLLM_GPU_UTIL:-0.28}
```

- [ ] **Step 2: Launch vLLM at 0.28, verify it loads and KV stays positive.**

```
wsl -d Ubuntu -e bash -c "bash /mnt/e/Claude/cosyvoice-local-tts/run_vllm_server.sh"
```
Expected: server reaches `/health`; startup log "Available KV cache memory" is **> 0 GiB**. If it crashes "No available memory for the cache blocks", set `0.30` and STOP (0.28 is below floor on this card) — record that and skip to Step 5.

- [ ] **Step 3: Synthesis sanity check.** With vLLM up, synthesize one en and one zh sentence (the existing `test_en.py` / `test_zh.py` in the cosyvoice repo, or a `scripts.measure` turn). Expected: clean audio, no garble, comparable TTFB to baseline.

- [ ] **Step 4: Measure.** Run the `nvidia-smi` query in the vLLM-only state → record `S1_vllm@0.28`. Gate: `S1_vllm - S1_vllm@0.28 >= 200 MiB` to count as a win. If a further step to `0.26` still keeps KV positive, repeat Steps 2–4 at `0.26` and keep the lowest passing value.

- [ ] **Step 5: Commit (cosyvoice repo).**

```bash
cd /e/Claude/cosyvoice-local-tts && git add run_vllm_server.sh && git commit -m "perf(vram): cap vLLM gpu_memory_utilization to reclaim KV headroom"
```

---

### Task 2: vLLM `max_model_len` cap (Workstream A2)

**Files:**
- Modify: `E:\Claude\cosyvoice-local-tts\CosyVoice\cosyvoice\cli\model.py:307-312` (add an env-gated `max_model_len`)

**Interfaces:**
- Consumes: `S1_vllm@util` from Task 1.
- Produces: `S1_vllm@util+maxlen`, used by Task 6.

- [ ] **Step 1: Add the env-gated cap to `EngineArgs`.** Replace the `_gpu_util`/`engine_args` block (lines 307–312) with:

```python
        _gpu_util = float(os.getenv("COSYVOICE_VLLM_GPU_UTIL", "0.3"))
        # VRAM trim: TTS sequences are short (one sentence of speech tokens), so the
        # default max_model_len (the Qwen2 base, tens of thousands of tokens) reserves
        # far more KV cache than CosyVoice ever uses. Cap it to shrink the KV reservation
        # with no quality loss for short prompts. 0/empty => leave vLLM's default.
        _max_len_env = os.getenv("COSYVOICE_VLLM_MAX_LEN", "").strip()
        _max_len = int(_max_len_env) if _max_len_env else None
        _engine_kwargs = dict(model=model_dir,
                              skip_tokenizer_init=True,
                              enable_prompt_embeds=True,
                              enforce_eager=_eager,
                              gpu_memory_utilization=_gpu_util)
        if _max_len is not None:
            _engine_kwargs["max_model_len"] = _max_len
        engine_args = EngineArgs(**_engine_kwargs)
```

- [ ] **Step 2: Add the export to the launch script.** In `run_vllm_server.sh`, near the util export, add:

```bash
# VRAM trim: cap vLLM's max sequence length. CosyVoice generates short per-sentence
# token streams, so a small cap shrinks the KV reservation. Empty => vLLM default.
export COSYVOICE_VLLM_MAX_LEN=${COSYVOICE_VLLM_MAX_LEN:-2048}
```

- [ ] **Step 3: Launch and verify no truncation on a long reply.** Restart vLLM. Synthesize a deliberately **long multi-sentence zh paragraph** and listen to the end. Expected: the full text is spoken — nothing cut off. If the tail is dropped, raise `COSYVOICE_VLLM_MAX_LEN` (e.g. 4096) and repeat. Also confirm the startup KV line stays positive.

- [ ] **Step 4: Measure.** `nvidia-smi` query in vLLM-only state → record `S1_vllm@util+maxlen`. Gate: a further `>= 200 MiB` drop vs Task 1, with no truncation.

- [ ] **Step 5: Commit (cosyvoice repo).**

```bash
cd /e/Claude/cosyvoice-local-tts && git add CosyVoice/cosyvoice/cli/model.py run_vllm_server.sh && git commit -m "perf(vram): env-gated vLLM max_model_len cap for short TTS prompts"
```

---

### Task 3: MuseTalk `empty_cache()` after warmup (Workstream B1)

**Files:**
- Modify: `local_services/musetalk_server/app.py` — `MuseTalkEngine.load()`, right after `self._warmup()` (line ~181)

**Interfaces:**
- Consumes: `S2_idle` baseline.
- Produces: `S2_idle@emptycache`, used by Task 6.

- [ ] **Step 1: Add the guarded cache release.** In `load()`, immediately after the `self._warmup()` call and before the final `logger.info(...)`, insert:

```python
        # VRAM trim: warmup ran dummy segments to pay one-time cuDNN/kernel/alloc costs;
        # that leaves reserved-but-unused blocks in PyTorch's caching allocator. Return
        # them to the driver so the idle footprint reflects the real working set (this is
        # where the measured ~8.7GB vs the model's ~4-6GB gap mostly hides). Best-effort:
        # an empty_cache failure must never block the server coming up.
        try:
            if self.torch.cuda.is_available():
                self.torch.cuda.synchronize()
                self.torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 -- cache release is best-effort
            logger.exception("empty_cache after warmup failed (non-fatal).")
```

- [ ] **Step 2: Restart MuseTalk, confirm it still warms up and renders.**

```
.\scripts\run.ps1
```
Expected: "MuseTalk warmup done…" then "MuseTalk ready" in the log; one `scripts.measure` turn renders lips normally (no regression in frames/segment or fps).

- [ ] **Step 3: Measure.** `nvidia-smi` query at the `+MuseTalk+pipeline` idle state → record `S2_idle@emptycache`. Gate: `S2_idle - S2_idle@emptycache >= 200 MiB` to count as a win (if < 200 MiB, the reserved blocks weren't the gap — keep the change anyway, it's harmless, but note it as noise in Task 6).

- [ ] **Step 4: Commit.**

```bash
git add local_services/musetalk_server/app.py
git commit -m "perf(vram): release MuseTalk warmup caching-allocator blocks after load"
```

---

### Task 4: MuseTalk batch-size peak (Workstream B2)

**Files:**
- (No code change — `MUSETALK_BATCH` is already an env knob, `app.py:71`, default 8.) Measurement + decision only; if adopted, document in `.env.example`.

**Interfaces:**
- Consumes: `S2_midturn` baseline + the `live_trail_baseline` (Task 0) as the regression guard.
- Produces: a kept/rejected decision for `MUSETALK_BATCH=4` + `S2_midturn@batch4`, used by Task 6.

- [ ] **Step 1: Run a turn at batch 4 with profiling on.** Set `MUSETALK_BATCH=4` and `MUSETALK_PROFILE=1` in the MuseTalk launch env (via `.env` / `run.ps1` propagation), restart, drive a turn:

```
python -m scripts.measure --offline-capture
```
Record the `[profile]` gpu/composite ms-per-segment from the server log.

- [ ] **Step 2: Measure the mid-turn peak.** While the turn renders, run the `nvidia-smi` query → record `S2_midturn@batch4`. Gate (VRAM): `S2_midturn - S2_midturn@batch4 >= 200 MiB`.

- [ ] **Step 3: Regression-check fps / trail.** Confirm the turn still renders at fps with no new stall, and the measured trail is no worse than `live_trail_baseline`. **Decision rule:** adopt `MUSETALK_BATCH=4` as the documented default **only if** Step 2 shows a real VRAM win **and** Step 3 shows no fps/trail regression. Otherwise keep 8 and record batch as "no usable win".

- [ ] **Step 4: If adopted, document the knob** in `.env.example` (one why-line: batch caps activation-memory peak, not compute; lower it to fit a tighter card at the cost of per-frame overhead). Then commit:

```bash
git add .env.example
git commit -m "docs(vram): document MUSETALK_BATCH as the activation-memory lever"
```
(If not adopted: no commit; the finding is recorded in Task 6.)

---

### Task 5: Lag revisit — bound the avatar `out_q`, measure `live` (Workstream C)

**Files:**
- Modify: `local_services/musetalk_server/app.py:548` (`out_q = asyncio.Queue(maxsize=600)` → env-gated, tighter)

**Interfaces:**
- Consumes: `live_trail_baseline` (Task 0).
- Produces: `live_trail_tuned` + the steady-vs-live verdict, used by Task 6.

- [ ] **Step 1: Make `out_q` maxsize an env knob.** Replace line 548:

```python
    # Bounded queue of rendered frames; the pump drains it at a STEADY fps. A SMALLER
    # cap is the documented SAFE lag lever for live mode: under GPU contention the render
    # skips stale frames instead of letting the lips fall arbitrarily far behind the voice.
    # Do NOT re-lock the voice to video (that froze it). 600 ~= 30s @20fps (effectively
    # unbounded); tighten via MUSETALK_OUT_Q for a shorter max trail.
    out_q: asyncio.Queue = asyncio.Queue(maxsize=int(os.getenv("MUSETALK_OUT_Q", "600")))
```

- [ ] **Step 2: Measure `live` at a tight cap.** Set `MUSETALK_SYNC_MODE=live` and `MUSETALK_OUT_Q=24` (~1.2s @20fps) in `.env`, restart the pipeline, run:

```
python -m scripts.measure --offline-capture
```
Read the lip-start-vs-voice and steady-state trail → record `live_trail_tuned`. Gate: trail is bounded (no longer grows on long replies) and lips don't visibly skip on every turn (if skipping is constant, raise `MUSETALK_OUT_Q` toward 48 and re-measure).

- [ ] **Step 3: Build the side-by-side for the user to judge.** Capture one `steady` turn and one tuned-`live` turn on the **real WebRTC delivery path** (open `/client/`, speak/probe a multi-sentence turn) — NOT only the offline capture (it bypasses the transport and can't fairly judge the live trail). Note both behaviors plainly: steady = synced start, brief pause under a long stall; tuned-live = voice instant, lips trail ~`live_trail_tuned`, bounded.

- [ ] **Step 4: Revert `.env` to the steady default.** `MUSETALK_SYNC_MODE=steady`. The default does **not** change in this task — the user decides in the review whether to adopt tuned-live.

- [ ] **Step 5: Commit (the knob only; default unchanged).**

```bash
git add local_services/musetalk_server/app.py
git commit -m "feat(avatar): MUSETALK_OUT_Q knob to bound the live-mode lip trail"
```

---

### Task 6: Write the measured before/after + knob docs (Deliverable)

**Files:**
- Modify: `docs/gpu-memory-notes.md` (append a measured before/after section)
- Modify: `CLAUDE.md` / `WORKFLOW.md §8` knob references (add the new/confirmed knobs)
- Delete: `docs/superpowers/vram-baseline.md` (folded in)

**Interfaces:**
- Consumes: every recorded number from Tasks 0–5.

- [ ] **Step 1: Append the before/after table to `docs/gpu-memory-notes.md`,** in its existing stop-and-diff voice. Real MiB only; mark any sub-200 MiB delta as "noise, not a win".

```markdown
## VRAM trim — measured before/after (YYYY-MM-DD, feat/offline-stt-sensevoice)

| Lever | State measured | Before | After | Reclaimed |
|-------|----------------|--------|-------|-----------|
| vLLM gpu_util 0.30->0.XX | vLLM-only | <S1_vllm> | <S1_vllm@util> | <d> |
| vLLM max_model_len cap | vLLM-only | <S1_vllm@util> | <…+maxlen> | <d> |
| MuseTalk empty_cache | +MuseTalk idle | <S2_idle> | <…@emptycache> | <d> |
| MUSETALK_BATCH 8->4 | mid-turn peak | <S2_midturn> | <…@batch4> | <d / "no usable win"> |
| **Total** | | | | **<sum> (~target ~2-3GB)** |

Lag (compute-bound, NOT VRAM): live trail baseline +<n>s -> tuned (MUSETALK_OUT_Q=24) +<n>s, bounded.
Steady remains default; tuned-live offered. If still unacceptable -> spec #2 (TensorRT).
```

- [ ] **Step 2: Add/confirm the knobs in `CLAUDE.md` + `WORKFLOW.md §8`:** `COSYVOICE_VLLM_GPU_UTIL` (new value + safe floor), `COSYVOICE_VLLM_MAX_LEN` (new), `MUSETALK_OUT_Q` (new, live lag lever), and `MUSETALK_BATCH` if adopted — each with its one-line why and failure mode.

- [ ] **Step 3: Remove the scratch baseline file.**

```bash
git rm docs/superpowers/vram-baseline.md
```

- [ ] **Step 4: Commit.**

```bash
git add docs/gpu-memory-notes.md CLAUDE.md WORKFLOW.md
git commit -m "docs(vram): record measured trim before/after + new knobs; fold baseline"
```

---

## Self-Review

**Spec coverage:**
- Workstream A1 (gpu_util) → Task 1 ✓; A2 (max_model_len) → Task 2 ✓.
- Workstream B1 (empty_cache) → Task 3 ✓; B2 (batch) → Task 4 ✓.
- Workstream C (live trail, out_q SAFE lever, steady-default kept) → Task 5 ✓.
- Deliverable (measured table in gpu-memory-notes.md, knob docs) → Task 6 ✓.
- Baseline/measurement-method prerequisite → Task 0 ✓.
- Out-of-scope (TensorRT, hardware, whisper decoder, CPU-offload) → not implemented, hand-off noted in Task 6 Step 1 ✓.

**Placeholder scan:** Numeric results are intentionally `<n>`/`<d>` — they are *measured outputs*, the deliverable of running the plan, not unspecified code. All code/edits and commands are concrete. No "TBD/add error handling/similar to Task N".

**Type/name consistency:** `S0_down/S1_vllm/S2_idle/S2_midturn/live_trail_baseline/live_trail_tuned` used consistently across Tasks 0–6. Env names consistent: `COSYVOICE_VLLM_GPU_UTIL`, `COSYVOICE_VLLM_MAX_LEN`, `MUSETALK_BATCH`, `MUSETALK_OUT_Q`, `MUSETALK_SYNC_MODE`. The `model.py` edit keeps `_gpu_util`/`_eager` names from the existing code.
