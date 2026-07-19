"""Stand-in for CosyVoice sharing the GPU: continuous fp16 matmuls that steal SM cycles
from MuseTalk's render, the same way vLLM bursts the card while streaming a reply. A COMPUTE
hog, not a memory hog. Run it alongside scripts/_drive_frames.py to load the render.
See docs/PROBLEMS-AND-FIXES.md P16.

*** THIS HOG CANNOT FORCE A RENDER UNDERFLOW -- do NOT use it as a positive control. ***
Re-measured 2026-07-17 at the live config (SIZE=512/fps=12): N=8192 steals NO more than
N=4096 (MuseTalk gpu 567ms vs 569ms per 8-frame seg -- identical), because under Windows
WDDM the two processes time-slice to a ~50% floor however heavy the other one is. At fps=12
even the PyTorch path then lands ~643ms/seg, still inside the 667ms budget -> drift stays
flat and this hog PASSES whatever you are testing. The docstring used to claim "4096 =
heavy, forces render < 12fps"; that was true in 2026-07-01's config and is false now.

To actually reproduce P16's length-scaling drift, tighten the BUDGET instead of raising the
load -- drive at fps=25 (budget 8/25 = 320ms/seg, under PyTorch's ~455ms):
  python -m scripts._drive_frames output/reply_concise.wav 25     # TRT=0 -> +4.04s @13.56s
                                                                  # TRT=1 -> +0.32s flat
This hog is still useful for measuring the render COST under contention (gpu-ms per segment
via MUSETALK_PROFILE=1) -- just not for producing a failure.

Run in the musetalk env; Ctrl-C / Stop-Process to stop:
  E:\\miniconda3\\envs\\musetalk\\python.exe -u scripts/_gpu_contention_hog.py [N]
N = matmul size (default 4096). Both 4096 and 8192 pin the card to 100% util; neither
starves the render at fps=12 (see above).
"""
import sys, time
import torch

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
ITERS = 120 if N >= 4096 else 40
d = torch.device("cuda")
a = torch.randn(N, N, device=d, dtype=torch.float16)
b = torch.randn(N, N, device=d, dtype=torch.float16)
print(f"gpu contention hog running (N={N}, iters/loop={ITERS})", flush=True)
while True:
    for _ in range(ITERS):
        a = (a @ b) * 0.0001 + 0.5
    torch.cuda.synchronize()
