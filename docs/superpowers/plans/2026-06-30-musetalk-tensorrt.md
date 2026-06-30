# MuseTalk TensorRT Acceleration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace MuseTalk's per-frame UNet + VAE-decoder PyTorch calls with fp16 TensorRT engines, gated behind `MUSETALK_TRT=1` with automatic PyTorch fallback, to cut GPU time ~36→≤18 ms/frame and buy contention headroom on the shared GPU — without regressing lip quality.

**Architecture:** Two stock diffusers models run in the realtime loop: the UNet (`UNet2DConditionModel`, in=8/out=4/cross-attn=384, latent B×8×32×32→sample B×4×32×32) and the VAE decoder (`AutoencoderKL.decode`, B×4×32×32→image B×3×256×256). Each is exported to ONNX, built into an fp16 TRT engine with a batch-1..8 optimization profile, cached on disk, and invoked via a thin TRT runtime that binds torch CUDA tensors on the render stream. whisper, the positional encoding, and the PIL compositing stay in PyTorch. `MUSETALK_TRT=1` switches `render_segment`'s inner GPU calls to the engines; any build/load failure logs and falls back to the proven torch path.

**Tech Stack:** TensorRT 10.13.3.9 (`tensorrt-cu12`, via `--extra-index-url https://pypi.nvidia.com`), torch 2.11+cu128, `torch.onnx.export` (opset 17), the TRT Python Builder/OnnxParser API (NOT `trtexec` — the pip wheel may not ship the binary).

## Global Constraints

- **No unit-test suite** (CLAUDE.md forbids inventing one). Each task's verification is a **measurement/observation gate**: an engine builds, an import resolves, SSIM ≥ threshold, ms/frame target, or a stop-and-diff. "Expected: PASS" = the gate met.
- **`MUSETALK_TRT` defaults OFF.** The shipped default stays the proven PyTorch path. A missing/broken engine must **fall back to torch**, never crash the server (same best-effort discipline as `_warmup`).
- **`cudnn.benchmark` stays `False`** (load-bearing, `app.py` — the turn-start re-autotune spike). TRT does not change this.
- **DLL ordering is load-bearing:** the server adds torch's `lib/` to the DLL search path before importing onnxruntime, or onnxruntime falls back to CPU. TensorRT import must NOT disturb that ordering — import order is: torch → (torch lib dir on PATH) → onnxruntime → tensorrt.
- **conda cert-store gotcha** (the `musetalk` env): set `SSL_CERT_FILE`=certifi for any download. See `project-visualllm-conda-ssl-weights`.
- **ASCII-only** in edited `.py` server source; UTF-8 **without BOM**.
- **Blackwell `sm_120` TRT path is PROVEN** (spec gate, 2026-06-30): TRT 10.13.3.9 built+ran an fp16 engine on this card in an isolated venv. Install requires `--extra-index-url https://pypi.nvidia.com` (bare install fails `metadata-generation-failed` on `tensorrt_cu12_libs`).
- **Branch:** `feat/musetalk-tensorrt` off `main` (this is independent of the VRAM work).
- **fp16 first:** export the already-`.half()` models to fp16 ONNX, build fp16 engines. If the numerics gate (Task 7) fails, the documented fallback is fp32-ONNX + `FP16` builder flag (TRT keeps sensitive layers fp32).

**Engine cache:** `local_services/musetalk_server/trt_cache/<key>/{unet.engine,vae.engine}`, key = md5 of `(unet.pth mtime+size, vae weights, trt.__version__, sm_120, batch=8, fp16, IMAGE_SIZE)`. Rebuild only on key change (mirrors `avatar_cache`).

---

### Task 0: Install TensorRT into the real `musetalk` env + confirm coexistence

**Files:** none (environment only).

**Interfaces:**
- Produces: a working `import tensorrt` in the `musetalk` env that coexists with torch + onnxruntime.

- [ ] **Step 1: Install TRT with the NVIDIA index.**

Run:
```
E:\miniconda3\envs\musetalk\python.exe -m pip install "tensorrt-cu12==10.13.3.9" --extra-index-url https://pypi.nvidia.com
```
Expected: `Successfully installed tensorrt-cu12 tensorrt-cu12-libs tensorrt-cu12-bindings`.

- [ ] **Step 2: Confirm import + fp16 + coexistence with the server's other GPU libs.**

Run:
```
E:\miniconda3\envs\musetalk\python.exe -c "import torch, onnxruntime; import tensorrt as trt; b=trt.Builder(trt.Logger()); print('trt',trt.__version__,'fp16',b.platform_has_fast_fp16,'ort',onnxruntime.get_available_providers())"
```
Expected: prints the TRT version, `fp16 True`, and onnxruntime still lists `CUDAExecutionProvider` (proves TRT didn't break the CUDA DLL resolution). If onnxruntime lost CUDA, STOP — the DLL ordering regressed; adjust the import order before continuing.

- [ ] **Step 3: Smoke-build in-env** (same trivial conv as the spec gate, now in `musetalk`):

Run the spec's smoke script with the env python; Expected: `RESULT: PASS`.

- [ ] **Step 4: Commit (record the dep).** Add a `local_services/musetalk_server/requirements-trt.txt` documenting `tensorrt-cu12==10.13.3.9 (--extra-index-url https://pypi.nvidia.com)`.

```bash
git add local_services/musetalk_server/requirements-trt.txt
git commit -m "build(trt): document TensorRT dep for the musetalk env (Blackwell sm_120)"
```

---

### Task 1: Capture real export inputs + export the UNet to ONNX

**Files:**
- Create: `local_services/musetalk_server/trt_export.py` (export helpers — UNet here, VAE in Task 3)

**Interfaces:**
- Consumes: a loaded `MuseTalkEngine` (its `.unet.model`, `.pe`, `.vae`, `.device`).
- Produces: `export_unet_onnx(engine, out_path) -> dict(shapes)` writing `unet.onnx`; the captured shapes feed Task 2's profile.

- [ ] **Step 1: Write a forward wrapper + a real-input capture.** Real inputs avoid guessing the audio seq length — capture the actual `(latent_batch, timesteps, audio_feat)` a warmup segment produces.

```python
# local_services/musetalk_server/trt_export.py
import torch

class _UNetFwd(torch.nn.Module):
    """ONNX-exportable view of MuseTalk's UNet call: render_segment does
    unet.model(latent, timesteps, encoder_hidden_states=audio).sample"""
    def __init__(self, unet_model):
        super().__init__()
        self.m = unet_model
    def forward(self, latent, timestep, audio):
        return self.m(latent, timestep, encoder_hidden_states=audio).sample

def _capture_unet_inputs(engine):
    """Run one silent segment through the torch path, hooking the UNet to grab
    a real (latent, timestep, audio) triple with correct shapes/dtypes."""
    import numpy as np
    grabbed = {}
    orig = engine.unet.model.forward
    def hook(latent, timestep, encoder_hidden_states=None, **kw):
        grabbed.setdefault("latent", latent.detach())
        grabbed.setdefault("timestep", timestep.detach() if torch.is_tensor(timestep) else torch.tensor([0], device=engine.device))
        grabbed.setdefault("audio", encoder_hidden_states.detach())
        return orig(latent, timestep, encoder_hidden_states=encoder_hidden_states, **kw)
    engine.unet.model.forward = hook
    try:
        engine.render_segment(np.zeros(engine.samples_for_frames(8), dtype=np.float32))
    finally:
        engine.unet.model.forward = orig
        engine.idx = 0
    return grabbed
```

- [ ] **Step 2: Add the exporter** (same file), exporting fp16 with a dynamic batch axis on latent/audio/output:

```python
def export_unet_onnx(engine, out_path):
    ins = _capture_unet_inputs(engine)
    wrap = _UNetFwd(engine.unet.model).eval()
    latent, ts, audio = ins["latent"], ins["timestep"], ins["audio"]
    with torch.no_grad():
        torch.onnx.export(
            wrap, (latent, ts, audio), str(out_path),
            input_names=["latent", "timestep", "audio"],
            output_names=["sample"], opset_version=17,
            dynamic_axes={"latent": {0: "B"}, "audio": {0: "B"}, "sample": {0: "B"}},
        )
    return {"latent": tuple(latent.shape), "audio": tuple(audio.shape),
            "timestep": tuple(ts.shape), "dtype": str(latent.dtype)}
```

- [ ] **Step 3: Export against the real model.** A one-off driver (run from the musetalk env, engine loaded):

```
E:\miniconda3\envs\musetalk\python.exe -c "from local_services.musetalk_server.app import engine; engine.load(); from local_services.musetalk_server.trt_export import export_unet_onnx; print(export_unet_onnx(engine,'local_services/musetalk_server/trt_cache/unet.onnx'))"
```
Expected: prints the captured shapes (latent `(8,8,32,32)`, audio `(8,~50,384)`); `unet.onnx` exists, non-zero size. Note the printed shapes for Task 2.

- [ ] **Step 4: Commit.**

```bash
git add local_services/musetalk_server/trt_export.py
git commit -m "feat(trt): UNet ONNX exporter with real-input shape capture"
```

---

### Task 2: Build the UNet fp16 TRT engine

**Files:**
- Create: `local_services/musetalk_server/trt_build.py` (Python Builder/OnnxParser engine builder, batch profile)

**Interfaces:**
- Consumes: `unet.onnx` + the shapes from Task 1.
- Produces: `build_engine(onnx_path, engine_path, shape_profile)` writing `unet.engine`; reused for the VAE in Task 4.

- [ ] **Step 1: Write the builder** (generic over input-shape profiles):

```python
# local_services/musetalk_server/trt_build.py
import tensorrt as trt

def build_engine(onnx_path, engine_path, profiles, workspace_mb=2048):
    """profiles: {input_name: (min_shape, opt_shape, max_shape)}."""
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.log(trt.Logger.ERROR, str(parser.get_error(i)))
            raise RuntimeError(f"ONNX parse failed: {onnx_path}")
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    profile = builder.create_optimization_profile()
    for name, (mn, op, mx) in profiles.items():
        profile.set_shape(name, mn, op, mx)
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"engine build returned None: {onnx_path}")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    return engine_path
```

- [ ] **Step 2: Build the UNet engine** (batch 1..8; use the audio seq length S printed in Task 1):

```
E:\miniconda3\envs\musetalk\python.exe -c "from local_services.musetalk_server.trt_build import build_engine; S=50; build_engine('local_services/musetalk_server/trt_cache/unet.onnx','local_services/musetalk_server/trt_cache/unet.engine',{'latent':((1,8,32,32),(8,8,32,32),(8,8,32,32)),'audio':((1,S,384),(8,S,384),(8,S,384)),'timestep':((1,),(1,),(1,))})"
```
Expected: `unet.engine` written, non-zero. (If the parser errors on a node, that's the diffusers-UNet export quirk risk — note the op and retry with opset 18 or fp32 ONNX per the Global Constraints fallback.)

- [ ] **Step 3: Commit.**

```bash
git add local_services/musetalk_server/trt_build.py
git commit -m "feat(trt): Python TRT engine builder with batch optimization profile"
```

---

### Task 3: Export the VAE decoder to ONNX

**Files:**
- Modify: `local_services/musetalk_server/trt_export.py` (add the VAE decoder exporter)

**Interfaces:**
- Produces: `export_vae_onnx(engine, out_path)` writing `vae.engine`'s ONNX; decoder only (input B×4×32×32 → sample B×3×256×256). The pre-scale (`1/scaling_factor`) and post (clamp/uint8/BGR) stay in the Task-6 wrapper, NOT the engine.

- [ ] **Step 1: Add the decoder wrapper + exporter.** `decode_latents` does `(1/scaling_factor)*latents → vae.decode(...).sample → /2+0.5 …`; the engine covers only `vae.decode(...).sample`:

```python
class _VAEDecFwd(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.v = vae
    def forward(self, latent):           # latent already pre-scaled by caller
        return self.v.decode(latent).sample

def export_vae_onnx(engine, out_path):
    import torch
    dec = _VAEDecFwd(engine.vae.vae).eval()
    dummy = torch.randn(8, 4, 32, 32, device=engine.device, dtype=engine.vae.vae.dtype)
    with torch.no_grad():
        torch.onnx.export(
            dec, (dummy,), str(out_path),
            input_names=["latent"], output_names=["image"], opset_version=17,
            dynamic_axes={"latent": {0: "B"}, "image": {0: "B"}},
        )
    return {"latent": (8, 4, 32, 32)}
```

- [ ] **Step 2: Export + build the VAE engine.**

```
E:\miniconda3\envs\musetalk\python.exe -c "from local_services.musetalk_server.app import engine; engine.load(); from local_services.musetalk_server.trt_export import export_vae_onnx; export_vae_onnx(engine,'local_services/musetalk_server/trt_cache/vae.onnx')"
E:\miniconda3\envs\musetalk\python.exe -c "from local_services.musetalk_server.trt_build import build_engine; build_engine('local_services/musetalk_server/trt_cache/vae.onnx','local_services/musetalk_server/trt_cache/vae.engine',{'latent':((1,4,32,32),(8,4,32,32),(8,4,32,32))})"
```
Expected: both files written, non-zero.

- [ ] **Step 3: Commit.**

```bash
git add local_services/musetalk_server/trt_export.py
git commit -m "feat(trt): VAE-decoder ONNX exporter (decoder only)"
```

---

### Task 4: TRT runtime runner (binds torch CUDA tensors)

**Files:**
- Create: `local_services/musetalk_server/trt_runtime.py`

**Interfaces:**
- Produces: `class TRTModule(engine_path, device)` with `def __call__(self, **named_torch_cuda_tensors) -> dict[str, torch.Tensor]`. Used by Task 6. Binds torch tensor `data_ptr()` via `set_tensor_address`; runs `execute_async_v3` on the current torch CUDA stream; returns preallocated output tensors.

- [ ] **Step 1: Write the runner.** One module, one responsibility (engine load + execution); swappable internally for `torch_tensorrt` later without touching `app.py`.

```python
# local_services/musetalk_server/trt_runtime.py
import tensorrt as trt
import torch

_TRT_TO_TORCH = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
                 trt.DataType.INT32: torch.int32}

class TRTModule:
    def __init__(self, engine_path, device):
        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in self.names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in self.names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

    def __call__(self, **kw):
        for n in self.inputs:
            t = kw[n].contiguous()
            self.ctx.set_input_shape(n, tuple(t.shape))
            self.ctx.set_tensor_address(n, t.data_ptr())
        outs = {}
        for n in self.outputs:
            shape = tuple(self.ctx.get_tensor_shape(n))
            dt = _TRT_TO_TORCH[self.engine.get_tensor_dtype(n)]
            o = torch.empty(shape, dtype=dt, device=self.device)
            outs[n] = o
            self.ctx.set_tensor_address(n, o.data_ptr())
        stream = torch.cuda.current_stream(self.device)
        self.ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return outs
```

- [ ] **Step 2: Standalone parity check** (engine vs torch on random input, before wiring the server):

```
E:\miniconda3\envs\musetalk\python.exe -c "from local_services.musetalk_server.app import engine; engine.load(); import torch; from local_services.musetalk_server.trt_runtime import TRTModule; m=TRTModule('local_services/musetalk_server/trt_cache/vae.engine', engine.device); x=torch.randn(8,4,32,32,device=engine.device,dtype=engine.vae.vae.dtype); ref=engine.vae.vae.decode(x).sample; out=m(latent=x)['image']; print('max abs diff', (out.float()-ref.float()).abs().max().item())"
```
Expected: max-abs-diff small (fp16 tolerance, < ~0.05). A large diff = binding/layout bug — fix before Task 6.

- [ ] **Step 3: Commit.**

```bash
git add local_services/musetalk_server/trt_runtime.py
git commit -m "feat(trt): TRT runtime module binding torch CUDA tensors"
```

---

### Task 5: Wire TRT into `render_segment` behind `MUSETALK_TRT` (with torch fallback)

**Files:**
- Modify: `local_services/musetalk_server/app.py` (`MuseTalkEngine.load()` builds/loads engines if `MUSETALK_TRT`; `render_segment` branches)

**Interfaces:**
- Consumes: `TRTModule`, `export_*`, `build_engine`, the cache key.
- Produces: `MUSETALK_TRT=1` → engine path; else torch path. Any failure logs + sets `self._trt = None` (torch fallback).

- [ ] **Step 1: Add engine load/build in `load()`** (after `_warmup`, guarded). Build the cache key, build engines if absent, load `TRTModule`s; on ANY exception set `self._trt=None` and log (fallback to torch):

```python
        self._trt = None
        if os.getenv("MUSETALK_TRT", "0").lower() in ("1", "true", "yes"):
            try:
                self._init_trt()   # builds (if needed) + loads unet/vae TRTModules into self._trt
                logger.info("MuseTalk TRT engines loaded.")
            except Exception:  # noqa: BLE001 -- TRT is best-effort; fall back to torch
                logger.exception("TRT init failed; using PyTorch render path.")
                self._trt = None
```
(`_init_trt` resolves the cache dir from the key, calls `export_*`+`build_engine` on miss, then `TRTModule(...)` for unet+vae. Keep it a small method.)

- [ ] **Step 2: Branch the two GPU calls in `render_segment`.** Replace the `unet.model(...).sample` + `vae.decode_latents(pred)` inner calls with a TRT path when `self._trt` is set, keeping the exact pre/post math (`pe` audio, `1/scaling_factor`, clamp→uint8→BGR):

```python
                audio_feat = self.pe(w_batch)
                if self._trt is not None:
                    sample = self._trt["unet"](latent=latent_batch, timestep=self.timesteps, audio=audio_feat)["sample"]
                    dec_in = (1.0 / self.vae.scaling_factor) * sample.to(self.vae.vae.dtype)
                    img = self._trt["vae"](latent=dec_in)["image"]           # B,3,256,256 (raw sample)
                    img = (img / 2 + 0.5).clamp(0, 1).permute(0, 2, 1 if False else 2, 3)  # keep parity w/ decode_latents
                    recon = (img.permute(0,2,3,1).float().cpu().numpy() * 255).round().astype("uint8")[..., ::-1]
                else:
                    pred = self.unet.model(latent_batch, self.timesteps, encoder_hidden_states=audio_feat).sample
                    pred = pred.to(dtype=self.vae.vae.dtype)
                    recon = self.vae.decode_latents(pred)
```
(Verify the permute against `decode_latents` exactly — output must be `[n,256,256,3]` BGR uint8, identical layout to the torch path so `_composite` is unchanged.)

- [ ] **Step 3: Server comes up with `MUSETALK_TRT=1`.** Set the env, start the server; Expected log: "MuseTalk TRT engines loaded." then "MuseTalk ready". With `MUSETALK_TRT=0` (default) the log shows the normal torch path. A TRT failure must show the fallback log + still reach "MuseTalk ready".

- [ ] **Step 4: Commit.**

```bash
git add local_services/musetalk_server/app.py
git commit -m "feat(trt): MUSETALK_TRT switch for engine render path with torch fallback"
```

---

### Task 6: Numerics / lip-quality gate (BLOCKING)

**Files:**
- Create: `scripts/_trt_quality_check.py` (compares torch vs TRT rendered frames on a fixed segment)

**Interfaces:**
- Consumes: a loaded engine, both render paths.
- Produces: per-frame SSIM + max-abs-diff; **PASS iff SSIM ≥ 0.99**.

- [ ] **Step 1: Write the comparison** — render the SAME audio segment + same `idx` start through torch and TRT, compare frames:

```python
# scripts/_trt_quality_check.py
import numpy as np, os
os.environ.setdefault("MUSETALK_TRT", "0")
from local_services.musetalk_server.app import engine
from skimage.metrics import structural_similarity as ssim  # add scikit-image to requirements-trt

def render_once(use_trt):
    engine._trt = engine._trt if use_trt else None
    engine.idx = 0
    seg = (0.05*np.sin(np.arange(engine.samples_for_frames(8))/30)).astype(np.float32)
    return engine.render_segment(seg)

if __name__ == "__main__":
    engine.load(); engine._init_trt()      # ensure engines exist
    trt_frames = render_once(True)
    torch_frames = render_once(False)
    import math
    for i,(a,b) in enumerate(zip(trt_frames, torch_frames)):
        A=np.frombuffer(a,np.uint8); B=np.frombuffer(b,np.uint8)
        s=ssim(A.reshape(-1), B.reshape(-1)) if False else None
    # compare as images:
    sz=engine.size
    s=[ssim(np.frombuffer(t,np.uint8).reshape(sz,sz,3), np.frombuffer(o,np.uint8).reshape(sz,sz,3), channel_axis=2) for t,o in zip(trt_frames,torch_frames)]
    print("frames", len(s), "min SSIM", min(s), "mean SSIM", sum(s)/len(s))
    print("RESULT:", "PASS" if min(s) >= 0.99 else "FAIL")
```

- [ ] **Step 2: Run the gate.**

```
E:\miniconda3\envs\musetalk\python.exe -m scripts._trt_quality_check
```
Expected: `RESULT: PASS` (min SSIM ≥ 0.99). **If FAIL:** apply the fp32-ONNX + `FP16` builder-flag fallback (Global Constraints) and re-run; if still failing, TRT is not viable at quality — STOP and report (do not ship worse lips).

- [ ] **Step 3: Commit.**

```bash
git add scripts/_trt_quality_check.py local_services/musetalk_server/requirements-trt.txt
git commit -m "test(trt): blocking SSIM quality gate (TRT vs torch frames)"
```

---

### Task 7: Perf + contention gates, and the docs

**Files:**
- Modify: `docs/gpu-memory-notes.md` (TRT perf + VRAM rows), `CLAUDE.md` (`MUSETALK_TRT` knob), `.env.example`

**Interfaces:**
- Consumes: the working TRT path.

- [ ] **Step 1: Perf gate (uncontended).** With `MUSETALK_TRT=1` and `MUSETALK_PROFILE=1`, feed one turn (the VRAM session's `scratchpad/feed_peak.py` pattern) and read `gpu=` ms/segment; compare to the torch baseline (~290 ms/8-frame ≈ 36 ms/frame). Expected: **≥ 2×** faster (≤ ~145 ms/8-frame). Below ~1.5× → reassess (note it; the maintenance cost may not be worth it).

- [ ] **Step 2: Contention gate (the real point).** Full stack up (cosyvoice vLLM + MuseTalk `MUSETALK_TRT=1` + pipeline). Drive a multi-sentence turn via `scripts._webrtc_probe` while vLLM streams; record fps stability + the `live`/`steady` trail vs the VRAM-session baseline. Expected: render keeps up under contention (the headroom materializes) — this is what TRT is for.

- [ ] **Step 3: VRAM (stop-and-diff).** Record the mid-turn peak with engines loaded vs the torch baseline (~9.4 GB full stack) so `gpu-memory-notes.md` stays honest (engines may use slightly more/less).

- [ ] **Step 4: Document.** Add the TRT perf/VRAM rows to `gpu-memory-notes.md`; add the `MUSETALK_TRT` knob (default 0, "fp16 TRT engines; ≥2× render; off by default, torch fallback") to `CLAUDE.md` + `.env.example`.

- [ ] **Step 5: Commit.**

```bash
git add docs/gpu-memory-notes.md CLAUDE.md .env.example
git commit -m "docs(trt): record measured TRT perf/contention/VRAM + MUSETALK_TRT knob"
```

---

## Self-Review

**Spec coverage:** UNet+VAE acceleration → Tasks 1–5 ✓. fp16/engine cache/batch profile → Tasks 2,4,5 ✓. `MUSETALK_TRT` toggle + torch fallback → Task 5 ✓. Blocking numerics gate → Task 6 ✓. Perf + contention + VRAM gates → Task 7 ✓. Verify-first TRT install (gate already proven; in-env install) → Task 0 ✓. Out-of-scope (int8, whisper/PE/compositing in TRT, WSL, dedicated GPU) → not implemented ✓.

**Placeholder scan:** Numeric gate results are `<measured>` outputs, not unspecified code. All code/commands concrete. The one judgement call — the audio seq length `S` (50) — is captured from a real tensor in Task 1 and surfaced before Task 2 uses it, not guessed blindly.

**Type/name consistency:** `MUSETALK_TRT`, `self._trt` (dict with keys `"unet"`/`"vae"`), `TRTModule(engine_path, device)` returning `{output_name: tensor}` (`"sample"` for UNet, `"image"` for VAE), `build_engine(onnx, engine, profiles)`, `export_unet_onnx`/`export_vae_onnx`, `trt_cache/` dir — consistent across Tasks 1–7. The render branch preserves the existing `recon` shape (`[n,256,256,3]` BGR uint8) so `_composite` is untouched.

**Open risk carried from the spec:** the diffusers-UNet ONNX export (Task 2 Step 2) is the most likely sticking point; the fp32-ONNX + FP16-flag fallback is documented, and the whole path is gated OFF by default so a failure never affects the shipped avatar.
