# MuseTalk TensorRT Acceleration — Design (spec #2)

**Date:** 2026-06-30
**Branch:** `feat/musetalk-tensorrt` (off `main`; touches only `local_services/musetalk_server/`)
**Goal:** TensorRT-accelerate MuseTalk's per-frame render (the UNet + VAE decoder) to cut GPU time
~**36 ms/frame → ~12–18 ms/frame**, buying **contention headroom** on the shared GPU — so the render
keeps up while vLLM streams a reply. This converts today's "steady feels good" into spare capacity for
higher fps/quality, a near-zero-trail `live`, or freeing SMs for faster zh TTS. **fp16 engines, with a
numerics/lip-quality gate vs the PyTorch path so quality cannot silently regress.**

> **Status: this is OPTIONAL headroom, not a pain-fix.** The parent VRAM/lag work (spec #1) already
> satisfied both goals — the stack fits ~8.4 GB and `steady` feels good on `/client`. The cheaper
> competing route is a **dedicated avatar GPU** (now viable — the avatar alone fits ~4.8 GB), which kills
> contention with zero render-code risk. TensorRT is the **single-card** way to get the same headroom.
> Build this only if a single-card smooth-at-higher-quality deploy is the target.

## Decisions (from brainstorming + the architecture probe)

| Question | Decision |
|----------|----------|
| What to accelerate | The **UNet** (`diffusers.UNet2DConditionModel`, SD-v1-4 arch) and the **VAE decoder** (`AutoencoderKL.decode`). These are the only GPU models in the realtime loop and the two most-exported architectures for TRT (every SD-on-TRT demo exports exactly them). |
| What to leave in PyTorch | **whisper-tiny** (4-layer encoder, ~6 ms/seg — not worth the risk), the **PositionalEncoding** (trivial), and the **compositing** (CPU/PIL `get_image_blending`, ~10 ms/frame — not GPU-bound; a later spec if needed). |
| Export route | **torch → ONNX (opset 17) → TRT engine (fp16)** via `trtexec`/`polygraphy`, plus a thin TRT-runtime runner binding torch CUDA tensors on the render stream. Try `torch_tensorrt` compile FIRST (far simpler integration if it works on this Windows env); fall back to ONNX→engine if it doesn't. |
| Precision | **fp16** (weights are already `.half()`). NOT int8 — quantization is a separate quality-risky spec. |
| Dynamic shapes | One engine per model with an **optimization profile covering batch 1..`MUSETALK_BATCH`(8)** — the turn-START segment and the `speech_end` tail are partial batches (the same shape variation that bit `cudnn.benchmark`; here it's handled by the profile, not re-autotune). |
| Toggle / fallback | New **`MUSETALK_TRT=1`** knob. Off by default. On → render uses the engines; a build/load failure logs and **falls back to the PyTorch path** (never blocks startup, like `_warmup`). |
| Engine cache | Cache built engines on disk keyed by `(model-weight hash, TRT version, GPU arch sm_120, batch, fp16)` — mirrors the existing `avatar_cache` materials pattern. Rebuild only on a key change. |

## ✅ Verify-FIRST gate — PASSED (2026-06-30)

**TensorRT must support the Blackwell 5060 Ti (`sm_120`).** Blackwell consumer support landed in
**TensorRT 10.7+**; an older TRT refuses to build engines for the card. **This gate was tested and
PASSED** before writing the plan:
- The card reports **`sm_120`**; the `musetalk` env is **torch 2.11.0+cu128, CUDA 12.8, Python 3.10**.
- **`tensorrt-cu12`** wheels are available up to **11.1** (the whole 10.7+→11.x Blackwell-capable range)
  for this interpreter. **Install requires `--extra-index-url https://pypi.nvidia.com`** — the bare
  `pip install tensorrt-cu12` fails `metadata-generation-failed` on `tensorrt_cu12_libs` (the libs come
  from NVIDIA's index, not PyPI). This is a known gotcha; bake it into the install task.
- Verified in an **isolated venv** (not the precious `musetalk` env): TRT **10.13.3.9** imported,
  `platform_has_fast_fp16=True`, and a trivial fp16 conv network **built + serialized (128 KB) +
  deserialized** — i.e. GPU codegen for `sm_120` succeeded. Smoke test: `scratchpad/trt_smoke.py`.

**Conclusion: the single-card from-scratch ONNX→TRT path is buildable on this box.** The WSL / dedicated-
GPU fallbacks are NOT needed. The implementation plan's Task 0 narrows to *installing TRT into the
real `musetalk` env* (with the NVIDIA index) and confirming it coexists with the env's torch/onnxruntime
+ the existing DLL-search-path ordering (torch `lib/` added before onnxruntime — must not regress).

## Architecture (what the realtime loop becomes)

Today, per `render_segment` (`musetalk_server/app.py`):
```
whisper_chunk -> pe() -> audio_feat
unet.model(latent_batch, timesteps=[0], encoder_hidden_states=audio_feat).sample  # GPU ~36ms/8-frame
  -> vae.decode_latents(pred)  # GPU, [n,256,256,3] BGR uint8
  -> _composite()              # CPU/PIL ~10ms/frame
```
With `MUSETALK_TRT=1`, the two GPU calls are replaced by engine invocations:
- **UNet engine** — inputs: `latent` (B,8,32,32 fp16), `timestep` (scalar, always 0), `encoder_hidden_states`
  (B,S,384 fp16 from `pe`); output: `sample` (B,4,32,32). (Confirm in/out channels against `musetalk.json`
  during Task 0 — MuseTalk concatenates masked+ref latents → 8-ch in, 4-ch out.)
- **VAE-decoder engine** — input: `latent` (B,4,32,32 fp16, pre-scaled by `1/scaling_factor` in the
  wrapper); output: image (B,3,256,256). The `decode_latents` post-step (clamp, →uint8, BGR) stays in the
  wrapper around the engine.
- A small **TRT runner** module (`musetalk_server/trt_runtime.py`) owns engine load, the execution context,
  the optimization profile, and binds torch CUDA tensors (zero-copy via DLPack / same stream). One module,
  one responsibility, swappable for `torch_tensorrt` internally without touching `app.py`.

`render_segment` gains a one-line branch: `self._render = self._render_trt if TRT else self._render_torch`.
Everything else (whisper, PE, compositing, the pump, A/V sync) is untouched.

## Validation (no unit-test suite here — these are the gates)

1. **Numerics / lip-quality gate (BLOCKING).** On a fixed reference turn, run the SAME latents+audio through
   both paths and compare the rendered frames: **per-frame max-abs-diff + SSIM**, plus an eyeball on the
   muxed mp4. fp16-TRT vs fp16-torch must be **visually identical** (SSIM ≥ ~0.99). A quality regression
   fails the spec — accelerating into worse lips is not a win.
2. **Perf gate.** `MUSETALK_PROFILE=1` gpu-ms/frame, torch vs TRT, on the same segment. **Target ≥ 2×**
   (≈36 ms → ≤18 ms). Below ~1.5× isn't worth the maintenance burden — stop and reassess.
3. **Contention gate (the actual point).** Full stack up; drive a multi-sentence turn while vLLM streams;
   measure the `live` trail and the `steady` pause vs the spec-#1 baseline. Confirm the headroom materializes
   under real contention (not just uncontended).
4. **VRAM (stop-and-diff).** TRT engines may use slightly more or less than torch fp16; record the mid-turn
   peak so `docs/gpu-memory-notes.md` stays honest.

## Out of scope
- **int8/fp8 quantization** (quality-risky; own spec if fp16 headroom is insufficient).
- **whisper-tiny and the PE** in TRT (negligible, high fiddle).
- **GPU compositing** (the ~10 ms/frame PIL blend; only if it becomes the new bottleneck after the UNet/VAE win).
- **Moving MuseTalk to WSL** as the *primary* design — considered (Linux TRT is better supported), but it's a
  much larger move; it's only the **Task-0-failure fallback**, not the plan.
- The **dedicated-GPU route** — the competing hardware option; tracked separately, not built here.

## Risks
- **Blackwell `sm_120` TRT support on Windows/cu128** — the gating unknown (Task 0). Mitigation: verify
  first; pivot to WSL/dedicated-GPU if it fails.
- **diffusers UNet ONNX export quirks** (cross-attention, timestep embedding) — known-but-fiddly; mitigated by
  the well-trodden SD-on-TRT recipe (NVIDIA TensorRT SD demo / StreamDiffusion as references) and trying
  `torch_tensorrt` first.
- **fp16 numerics drift** → the BLOCKING quality gate catches it; if it fails, selectively keep sensitive
  layers in fp32 (mixed precision) or fall back.
- **conda cert-store gotcha** (the `musetalk` env) may bite TRT/pip downloads — apply the known
  `SSL_CERT_FILE`=certifi fix (see `project-visualllm-conda-ssl-weights`).
- **Maintenance / fragility** — engines are arch+TRT-version specific; the cache key + the PyTorch fallback
  keep a broken/absent engine from bricking the avatar. `MUSETALK_TRT` defaults OFF so the shipped default
  is always the proven torch path.

## References
- TMElyralab/MuseTalk (arch: SD-v1-4 UNet + frozen VAE + whisper-tiny; 30 fps+ on V100).
- NVIDIA TensorRT Stable-Diffusion demo + StreamDiffusion — the UNet+VAE TRT engine-build recipe to follow.
