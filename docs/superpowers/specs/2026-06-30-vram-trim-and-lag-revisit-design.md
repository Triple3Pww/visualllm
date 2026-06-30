# VRAM Footprint Trim + Measured Lag Revisit — Design

**Date:** 2026-06-30
**Branch:** `feat/vram-trim` (off `main`; touches this repo + the sibling `E:\Claude\cosyvoice-local-tts` repo)
**Goal:** Claw back the *realistic* VRAM headroom on the shared 16 GB card via **reversible** knobs —
each validated by **stop-and-diff measurement**, not estimates — and make a **measured** decision on
the ~4 s avatar lag via `live` mode. **Steady stays the default.** This is spec **#1 of 2**; TensorRT
MuseTalk is the committed follow-up (spec #2), kept out of this spec so the safe, shippable win is
isolated from the risky one.

## Decisions (from brainstorming)

| Question | Decision |
|----------|----------|
| What is the goal? | **Shrink the footprint as far as it'll go (hardware-agnostic) + fix the ~4 s avatar lag.** No specific target card chosen yet. |
| Approach | **① Measured trim now + revisit `live` for the lag; ② TensorRT MuseTalk as a separate follow-up spec.** Sequenced — ship the safe win first, then attempt the risky one with a real baseline. |
| Lag default | **Keep `MUSETALK_SYNC_MODE=steady` as the default.** Tuned-`live` is *offered* for the user to judge, never flipped unilaterally. |
| Accepted ceiling | Software-only realistically reclaims **~2–3 GB** (→ a ~12 GB-class single-card fit), **not** 8 GB on one card. The two model weights have a real floor; the bigger win is deferred to spec #2 (TensorRT) / hardware. |
| Measurement method | **Stop-and-diff** on `nvidia-smi memory.used` — the only reliable attribution on this WDDM consumer card (per-process `used_memory` reads `[N/A]`). Established practice in `docs/gpu-memory-notes.md`. |

## The constraint that bounds this work (measured, authoritative)

From `docs/gpu-memory-notes.md` (stop-and-diff, 2026-06-30):

| Component | Now | Floor | Lever |
|-----------|-----|-------|-------|
| CosyVoice vLLM | ~6.0 GB | ~4.5 GB | `COSYVOICE_VLLM_GPU_UTIL` 0.30→~0.26 + `--max-model-len` cap. vLLM's own ~4 GB is a hard floor. |
| MuseTalk | ~8.7 GB | ~5 GB | `empty_cache()` after warmup + lower `MUSETALK_BATCH` peak. 8.7 vs the docstring's "4–6 GB" is mostly PyTorch *reserved/cached* + cuDNN kernels + batch-8 activations, **not** weights. |
| Pipeline | ~0 GB | ~0 GB | CPU-only (WebRTC + resampling). Nothing to do. |
| Windows desktop | ~0.9 GB | — | Dead end (closing apps freed ~168 MiB; not worth it). |

**Two facts that shape the design:** (1) The stack genuinely needs ~14.7 GB; a clean restart reclaims
nothing — there is no leak to claw back, only *reserved-but-unused* blocks + KV/activation headroom.
(2) **The lag is COMPUTE contention, not VRAM** (GPU sits at ~1 % between turns; vLLM and MuseTalk fight
for SMs *during* a turn). **Freeing VRAM does not touch the lag** — the lag is a separate axis, handled
only in Workstream C (and fully only by spec #2 / hardware).

Whisper is **`whisper-tiny`** (151 MB fp32 → ~75 MB fp16) — confirmed from its `config.json`. It is
**not** a lever; dropping its 4-layer decoder saves ~30 MB. Explicitly ignore it.

## Workstreams

Each change is reversible and `.env`/arg-driven. Each is gated on its own stop-and-diff measurement;
a change that doesn't measurably help (or regresses) is reverted, not kept "to be safe."

### A. TTS / vLLM trim — repo `E:\Claude\cosyvoice-local-tts`

- **A1 — `COSYVOICE_VLLM_GPU_UTIL` step-down.** Default 0.30 (≈4.8 GB). Step **0.30 → 0.28 → 0.26**,
  stopping at the lowest where the vLLM startup log's **"Available KV cache memory"** stays **positive**
  *and* a zh+en test sentence still synthesizes clean. `0.20` already crashed ("No available memory for
  the cache blocks"), so the safe floor is ~0.25 — do not go below without the log confirming.
- **A2 — `--max-model-len` cap.** The vLLM engine currently runs without an explicit `--max-model-len`
  (the `run_vllm_server.sh` launch uses defaults; the value lives in `tts_engine.py`/`app.py`). TTS
  sequences are short, so a smaller `max-model-len` shrinks the KV-cache reservation with no quality
  loss. Find the current effective value, cap it conservatively (measured-max + margin), then **verify a
  long reply isn't truncated** (a multi-sentence zh turn).
- **Validate A:** stop-and-diff `memory.used` in the **"vLLM loaded, MuseTalk/pipeline down"** state,
  before/after each of A1 and A2 (so each lever's gain is attributed separately).
- **Order interaction:** both interact with the load-order rule (**start CosyVoice before MuseTalk**).
  Change **one knob at a time** and re-check the vLLM startup log each time.

### B. MuseTalk trim — `local_services/musetalk_server/app.py`

- **B1 — `torch.cuda.empty_cache()` after `_warmup()`.** Warmup runs two dummy segments to pay
  one-time costs; it leaves reserved-but-unused blocks in PyTorch's caching allocator. An `empty_cache()`
  after warmup (and `cuda.synchronize()`) returns them to the driver. This is the most likely source of
  the 8.7 GB vs "4–6 GB" gap. Best-effort + guarded (must never break startup, like the warmup itself).
- **B2 — lower `MUSETALK_BATCH` peak (8 → 4), keep as a knob.** Batch size drives *activation* memory
  (the UNet/VAE forward holds `BATCH_SIZE` latents + features at once), not total compute. Lowering it
  cuts the **mid-turn peak** `memory.used`. Adopt the lower value **only if** fps and the C-measured
  trail do **not** regress (smaller batches = more Python/launch overhead per frame — watch
  `MUSETALK_PROFILE=1` gpu/composite times). Stays env-overridable either way.
- **Validate B:** stop-and-diff at the **"+MuseTalk fresh"** state (B1) **and** the **mid-turn peak**
  during a real turn (B2), before/after.

### C. Lag revisit (measured — `live` offered, steady kept default)

- **C1 — measure the current `live` trail.** Run `python -m scripts.measure --offline-capture` (and the
  WebRTC probe) in `MUSETALK_SYNC_MODE=live` to get the *actual* lip-trail number under current load
  (the "~0.75 s" figure is from earlier; re-measure on today's stack).
- **C2 — apply the documented SAFE lever only: bound the avatar server `out_q`** (drop stale frames so
  the lips can't fall arbitrarily far behind). The queue is already `maxsize=600`; tighten it so under
  contention the render *skips* rather than accumulates a backlog. **Never re-lock the voice to video**
  (fully-locked/video-master sync froze the voice on a render stall — confirmed; this is the one thing
  the design must not do). Re-measure the trail.
- **C3 — present steady vs tuned-`live` for the user to judge.** Deliver both as a side-by-side the user
  can watch on the real WebRTC delivery path (offline capture bypasses the transport and is not a fair
  judge of the live trail). **Do not change the default** — steady stays default; live is adopted only
  if the user explicitly prefers the tuned result.

## Deliverable

- **Real before/after table appended to `docs/gpu-memory-notes.md`**, in its existing
  "stop-and-diff authoritative" voice — measured MiB per lever, no estimates. Each kept knob documented
  with its safe floor + the failure mode that bounds it.
- Every adopted change reflected in `.env.example` / `CLAUDE.md` / `WORKFLOW.md §8` knob references,
  matching the existing why-line convention.
- A short verdict line on the lag (tuned-`live` trail number + the user's steady-vs-live call), and an
  explicit hand-off note to spec #2 if `live` is still unacceptable.

## Prerequisites

- The **full stack must be running** to measure (load order: cosyvoice vLLM → `scripts/run.ps1`).
  Measurement is the gate on every change, so the implementation session needs the GPU free and the
  stack up.
- On WDDM, per-process bytes read `[N/A]` (driver limitation, Windows **and** WSL `nvidia-smi`), so
  stop-and-diff is the only attribution method — budget for the stop/start cycles.

## Out of scope (→ spec #2, committed follow-up)

- **TensorRT MuseTalk** — the one move that fixes footprint *and* lag together (faster render = less
  activation VRAM + keeps up under contention, no `live` trail). High effort on this Windows/conda cu128
  stack; isolated into its own spec so it can fail without endangering spec #1's shipped win.
- Adding / dedicating a second GPU; cloud/remote avatar GPU. (The notes' own "real fix" — needs hardware.)
- Dropping Whisper's decoder (~30 MB; not worth the code risk).
- CPU-offloading TTS/avatar weights (not viable for real-time — pages over PCIe).

## Risks / notes (all reversible)

- **vLLM util too low →** "No available memory for the cache blocks" crash on load. Mitigation: step
  down 0.02 at a time, confirm the KV-cache log line stays positive, revert the knob if it crashes.
- **`max-model-len` too low →** truncated long TTS sentences. Mitigation: cap conservatively and test a
  long multi-sentence zh reply before adopting.
- **`empty_cache()` →** marginally slower next allocation (negligible; it happens once after warmup,
  between turns). Guard it so a failure never blocks startup.
- **Lower batch →** possible per-frame overhead / fps regression. Mitigation: gate adoption on the
  C-measured trail + `MUSETALK_PROFILE=1` timings; keep it a knob, don't hardcode.
- **`live` trail may still be unacceptable →** then the lag is genuinely a TensorRT/hardware problem.
  That is a *clean* outcome: spec #1 still ships the measured VRAM win, and the lag hands off to spec #2.
- **Measurement noise:** desktop VRAM drifts ~±170 MiB; treat sub-200 MiB "gains" as noise, not wins.
