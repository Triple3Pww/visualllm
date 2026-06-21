"""Offline Ditto render: audio + portrait -> photoreal talking-head mp4.

Unlike the realtime server (which streams frames over a websocket and trades motion
for latency), this drives Ditto's OFFLINE pipeline: it processes the whole clip at
once with full-sequence smoothing, so the motion is smoother and there is no
keep-up/drift pressure. It reuses the SAME pytorch checkpoints and CUDA-DLL fix as
the server (no TensorRT / C compiler needed on this box).

Runs in the dedicated 'ditto' conda env (the avatar models live there):
    conda run -n ditto python -m local_services.ditto_offline \
        --audio output/what_is_ai.wav --image assets/avatar.png \
        --out output/what_is_ai.mp4

The GPU is shared and single-tenant -- stop the realtime ditto_server first so the
two don't contend for VRAM.
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent / "ditto_server"
VENDOR = SERVER_DIR / "vendor" / "ditto-talkinghead"
CKPT = VENDOR / "checkpoints"
DATA_ROOT = str(CKPT / "ditto_pytorch")
CFG_PKL = str(CKPT / "ditto_cfg" / "v0.4_hubert_cfg_pytorch.pkl")

if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

# onnxruntime-gpu needs torch's CUDA-12 DLLs on the search path BEFORE it imports,
# else it silently falls back to CPU (~5x slower). Same fix as ditto_server/app.py.
try:
    import torch as _torch

    os.add_dll_directory(str(Path(_torch.__file__).parent / "lib"))
except Exception:  # noqa: BLE001
    pass


def render(audio_path: Path, image_path: Path, out_path: Path,
           steps: int = 50, overlap: int = 10) -> Path:
    import librosa
    from stream_pipeline_offline import StreamSDK

    # Resolve all paths to absolute BEFORE the chdir below, or the relative ones
    # (e.g. assets/avatar.png) would resolve against VENDOR and 404.
    audio_path = audio_path.resolve()
    image_path = image_path.resolve()
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Match the offline smoke's cwd so any cwd-relative model path inside Ditto
    # resolves the same way the server's load did.
    os.chdir(VENDOR)

    sdk = StreamSDK(CFG_PKL, DATA_ROOT)
    # steps = diffusion sampling timesteps (offline default 50 = best motion quality;
    # latency is irrelevant offline). overlap is the diffusion window granularity.
    sdk.setup(
        str(image_path), str(out_path),
        online_mode=False,
        sampling_timesteps=steps,
        overlap_v2=overlap,
        N_d=-1,
    )

    audio, _ = librosa.core.load(str(audio_path), sr=16000)
    num_f = math.ceil(len(audio) / 16000 * 25)  # 25 fps motion
    # A short fade in/out so the eyes/mouth open and settle cleanly at the ends.
    sdk.setup_Nd(N_d=num_f, fade_in=8, fade_out=8)

    # Offline: extract features for the whole clip and hand it to the diffuser at once.
    aud_feat = sdk.wav2feat.wav2feat(audio)
    sdk.audio2motion_queue.put(aud_feat)
    sdk.close()  # joins workers, finalizes the silent video at tmp_output_path

    # Mux the original voice onto the silent render; re-encode to yuv420p H.264 so it
    # plays everywhere (browsers / QuickTime / Windows).
    tmp = sdk.tmp_output_path
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y",
         "-i", tmp, "-i", str(audio_path),
         "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
         "-c:a", "aac", "-b:a", "192k", "-shortest",
         str(out_path)],
        check=True,
    )
    try:
        os.remove(tmp)
    except OSError:
        pass
    print(f"WROTE {out_path}  ({num_f} frames @25fps, {num_f / 25:.1f}s)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="input voice wav/mp3 (16k mono ok)")
    ap.add_argument("--image", default="assets/avatar.png", help="portrait png")
    ap.add_argument("--out", required=True, help="output mp4 path")
    ap.add_argument("--steps", type=int, default=50, help="diffusion steps (motion quality)")
    ap.add_argument("--overlap", type=int, default=10, help="diffusion window overlap")
    args = ap.parse_args()
    render(Path(args.audio), Path(args.image), Path(args.out),
           steps=args.steps, overlap=args.overlap)


if __name__ == "__main__":
    main()
