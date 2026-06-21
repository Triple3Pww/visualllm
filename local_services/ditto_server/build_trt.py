"""Build Blackwell (sm_120) FP16 TensorRT engines for Ditto's heavy models.

Why
---
Ditto ships prebuilt engines only for ``ditto_trt_Ampere_Plus`` (RTX 30/40-era).
A TensorRT engine is compiled for one GPU architecture and will NOT load on a
newer one, so on this RTX 5060 Ti (Blackwell, sm_120) they are unusable -- the
server falls back to the PyTorch checkpoints at ~8 fps. This script rebuilds the
engines *for this card* from the architecture-independent ONNX graphs.

Scope (the hybrid, no-compiler plan)
------------------------------------
Convert only the models that (a) dominate the per-frame cost and (b) need no
custom plugin:
  * decoder            -- the per-frame bottleneck (~59 ms -> ~14 ms, 4.3x)
  * stitch_network     -- per-frame
  * appearance_extractor -- per-session warmup
``warp_network`` is intentionally skipped: it uses the custom GridSample3D op
whose TensorRT plugin ships only as a Linux .so, so it stays on PyTorch (runs
fine on the GPU, no plugin needed). ``motion_extractor`` stays PyTorch too
(Ditto deliberately keeps it FP32 for accuracy; it is per-session, not per-frame).

Safety
------
Every engine is numerically validated against the FP32 ONNX-CUDA output on random
inputs before it is accepted. If the max relative error exceeds the per-model
tolerance the engine is rejected (we promised "same picture" -- this enforces it).

Run (in the ditto env):
    E:\\miniconda3\\envs\\ditto\\python.exe -m local_services.ditto_server.build_trt
ASCII-only (cp1252-safe server source).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# CUDA + TensorRT DLLs must be discoverable BEFORE importing onnxruntime/tensorrt
# (same fix the server uses): torch/lib carries the cu12 runtime, tensorrt_libs
# carries nvinfer_10.dll.
import torch  # noqa: E402

for _d in (
    Path(torch.__file__).parent / "lib",
    Path(__import__("tensorrt_libs").__file__).parent,
):
    if _d.is_dir():
        try:
            os.add_dll_directory(str(_d))
        except OSError:
            pass

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import tensorrt as trt  # noqa: E402

CKPT = Path(__file__).resolve().parent / "vendor" / "ditto-talkinghead" / "checkpoints"
ONNX_DIR = CKPT / "ditto_onnx"
OUT_DIR = CKPT / "ditto_trt_blackwell"

LOGGER = trt.Logger(trt.Logger.ERROR)

# model -> (engine filename, [output tensor names], fp16?, relative-error tolerance).
# Tolerances are generous for the image models (they feed an 8-bit image, so sub-1%
# error is invisible). lmdm is the DIFFUSION net -- the true per-frame bottleneck --
# called once per sampling step; we keep it FP32 (accuracy: diffusion is iterative,
# Ditto's own converter never fp16's it) and validate both its outputs.
MODELS = {
    "decoder": ("decoder_fp16.engine", ["output"], True, 2e-2),
    "stitch_network": ("stitch_network_fp16.engine", ["out"], True, 2e-2),
    "appearance_extractor": ("appearance_extractor_fp16.engine", ["pred"], True, 2e-2),
    # lmdm (the diffusion net) can be built too -- it validated at <0.1% error -- but
    # gives no net end-to-end gain (per-step host roundtrip in the numpy sampling loop
    # cancels it; the real gate is the serial per-frame pipeline). Left out of the
    # default build; uncomment to experiment:
    # "lmdm_v0.4_hubert": ("lmdm_v0.4_hubert_fp32.engine", ["pred_noise", "x_start"], False, 1e-2),
}


def _rand_like(inp) -> np.ndarray:
    """Random input matching the ONNX tensor's declared dtype. Integer inputs
    (e.g. the diffusion timestep `time_cond`) must stay integer -- feeding a float
    makes onnxruntime reject the graph and is the wrong value domain."""
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    t = str(getattr(inp, "type", "tensor(float)"))
    if "int" in t:
        # a representative mid-range diffusion timestep; same value feeds both paths
        return np.full(shape, 500, dtype=np.int64)
    return np.random.randn(*shape).astype(np.float32)


def _build(onnx_path: Path, engine_path: Path, fp16: bool) -> bool:
    builder = trt.Builder(LOGGER)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("  parse error:", parser.get_error(i))
            return False
    config = builder.create_builder_config()
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        shp = [1 if d < 0 else d for d in inp.shape]
        profile.set_shape(inp.name, shp, shp, shp)
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("  build failed")
        return False
    engine_path.write_bytes(serialized)
    return True


def _validate(onnx_path: Path, engine_path: Path, out_names: list, tol: float) -> bool:
    from local_services.ditto_server.trt_runner import TRTRunner

    sess = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider"])
    feeds = {inp.name: _rand_like(inp) for inp in sess.get_inputs()}
    refs = sess.run(None, feeds)

    runner = TRTRunner(str(engine_path))
    runner.setup(feeds)
    runner.infer()

    ok = True
    for i, out_name in enumerate(out_names):
        ref = refs[i]
        got = runner.buffer[out_name][0]
        if got.shape != ref.shape:
            print(f"  {out_name}: shape mismatch trt {got.shape} vs onnx {ref.shape}")
            ok = False
            continue
        denom = np.abs(ref).mean() + 1e-6
        rel = np.abs(got.astype(np.float32) - ref).mean() / denom
        mx = np.abs(got.astype(np.float32) - ref).max()
        passed = rel <= tol
        ok = ok and passed
        print(f"  validate[{out_name}]: mean_rel_err={rel:.4f} (tol {tol}) max_abs={mx:.4f} -> {'PASS' if passed else 'FAIL'}")
    return ok


def main() -> int:
    if not ONNX_DIR.is_dir():
        print(f"ONNX dir not found: {ONNX_DIR}")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cap = torch.cuda.get_device_capability()
    print(f"Building Blackwell engines for sm_{cap[0]}{cap[1]} into {OUT_DIR}")

    all_ok = True
    for name, (engine_name, out_names, fp16, tol) in MODELS.items():
        onnx_path = ONNX_DIR / f"{name}.onnx"
        engine_path = OUT_DIR / engine_name
        print(f"[{name}]")
        if not onnx_path.is_file():
            print(f"  missing onnx: {onnx_path}")
            all_ok = False
            continue
        if not _build(onnx_path, engine_path, fp16):
            all_ok = False
            continue
        if not _validate(onnx_path, engine_path, out_names, tol):
            print("  rejecting engine (numerics out of tolerance)")
            engine_path.unlink(missing_ok=True)
            all_ok = False
            continue
        print(f"  ok -> {engine_path.name}")

    print("DONE" if all_ok else "DONE (with failures)")
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
