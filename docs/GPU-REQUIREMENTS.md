# GPU requirements — what cards can run this system

_Researched 2026-07-15 (24th session, right after the P50 VRAM squeeze). Grounded in this repo's own
measurements (P16/P36/P50) plus web-verified vendor docs and July-2026 market prices. Re-check the
market numbers before buying; the software floors are stable._

## TL;DR

- **NVIDIA only**, compute capability **7.5+ on paper (RTX 20-series), 8.0+ in practice (RTX 30-series)**.
- **VRAM floor is now 8GB** (post-P50: whole project ≈ 6.9GB live). 16GB stays comfortable.
- **The real bar is compute under contention, not VRAM**: MuseTalk must hold ~12fps at
  `MUSETALK_SIZE=512` *while* vLLM (CosyVoice) bursts on the same card. The RTX 5060 Ti 16GB is the
  verified baseline and sits right at that edge (11.8–12fps; P36).
- Best upgrade per baht: **a second cheap NVIDIA card as a dedicated avatar GPU** — the structural
  contention fix STATUS/CLAUDE.md have pointed at since P20.

## Why each requirement exists

Three GPU consumers share one card: CosyVoice's autoregressive LM on **vLLM** (WSL2), the **MuseTalk**
render through **TensorRT**, and the Whisper/VAE feature extractors (torch CUDA). Each imposes a floor:

| Requirement | Floor | Why |
|---|---|---|
| Vendor | NVIDIA | vLLM + TensorRT + torch-CUDA paths; no ROCm/oneAPI ports here. (Sherpa STT is CPU — irrelevant.) |
| Compute capability | ≥ 7.5 (TRT 10's hard floor; Volta deprecated in TRT 10.0). vLLM: ≥ 7.0, but bfloat16/FP8 need ≥ 8.0 | Below 7.5 the TRT engines don't run at all → PyTorch fallback. On *this* 5060 Ti that fallback still clears the 12 fps budget, but by only **4 %** under contention (1.04×, vs TRT's 2.05× — re-measured 2026-07-17, P16); a card slower than this one has no margin left, so treat the fallback as non-viable rather than "slow but OK". |
| VRAM | ≥ 8GB | P50 squeeze: vLLM 2.3GB (`COSYVOICE_VLLM_GPU_UTIL=0.07` on 16GB; use **~0.14 on an 8GB card** — the knob is fraction-of-card) + avatar ~3.3GB (`MUSETALK_FREE_TORCH=1`) ≈ 6.9GB project share. |
| Compute throughput | ~RTX 5060 Ti class for the single-GPU 512px config | MuseTalk must render ≥ 12fps at 512px **under live CosyVoice contention** or steady-mode voice lags (P36: 768/1024 profiled fine in isolation, collapsed to ~10fps live). Slower cards likely need `MUSETALK_SIZE=256`. |
| OS | Windows + WSL2 (or Linux) | vLLM is Linux-only; we run it in WSL2. Any card with current drivers works under WSL2 CUDA. |

## Compute-capability matrix

| Generation | CC | Example cards | Runs this stack? |
|---|---|---|---|
| Maxwell / Pascal | 5.x–6.x | GTX 900 / 10-series | ❌ Neither vLLM nor TRT 10 |
| Volta | 7.0 | V100 | ⚠️ vLLM only; TRT 10 dropped it |
| Turing | 7.5 | RTX 20-series | ⚠️ Both run; fp16 only (no bf16/FP8), older attention kernels. GTX 16-series is 7.5 but has **no tensor cores** — starts, but far too slow |
| Ampere | 8.0/8.6 | RTX 30-series, A100 | ✅ Full support — the practical minimum |
| Ada | 8.9 | RTX 40-series | ✅ Full support |
| Blackwell | 12.0 | RTX 50-series (current card) | ✅ Full support (minus the FP8-conv TRT gap — the P-notes' FP8 dead end, which we don't use anyway) |

**TRT engines are compiled per-GPU** — `musetalk_server/trt_cache/` does not transfer across cards
(or major driver bumps). Any card change = rerun `trt_build.py` (scripted, one-time).

## GPU tiers (July 2026)

| Tier | Card | Verdict |
|---|---|---|
| Minimum viable (deploy target) | RTX 3060 Ti / 4060 / 3070 (8GB) | Fits VRAM post-P50 (`VLLM_GPU_UTIL≈0.14`), but compute is below the 5060 Ti → expect `MUSETALK_SIZE=256` to hold 12fps under contention. **Unverified — needs one live-eye pass on real 8GB hardware before claiming it.** |
| **Verified baseline** | **RTX 5060 Ti 16GB (current)** | Known-good. 512px is the measured ceiling (P36). |
| Single-GPU upgrade | RTX 5070 Ti 16GB (~$750) / 5080 16GB | ~1.7–2× compute → likely unlocks 768px and removes the steady-mode contention lag. Same VRAM. |
| End-game single card | RTX 5090 32GB (~$2,000+) | Solves compute + VRAM headroom at once; overkill for the 7GB footprint. |
| **Structural fix** | **Second GPU, dedicated avatar** | vLLM and MuseTalk are already separate processes → just `CUDA_VISIBLE_DEVICES` per process + a TRT rebuild, zero sync-code changes. The avatar alone renders ~31fps in isolation on the current card, so even another 5060 Ti (~$430) or a used 3070/4060 Ti clears the bar. Removes the P20/P36 bottleneck outright. |

## Market notes (July 2026 — re-verify before buying)

- **Used RTX 3090 24GB is no longer the cheap-VRAM play**: AI demand pushed used prices back to
  ~$1,050–1,250 — worse value than a new 5070 Ti.
- **Don't wait for the 50-series Super refresh** (5070 Ti Super 24GB, rumored $749–799): reportedly on
  indefinite hold / slipped toward late 2026–CES 2027 because datacenter demand is eating GDDR7 supply —
  and the same shortage is pushing consumer GPU prices *up*, not down.

## Sources

- vLLM GPU docs / compatibility: <https://docs.vllm.ai/en/stable/getting_started/installation/gpu/>,
  <https://www.speediyo.com/ai-infra/vllm-gpu-compute-capability-matrix>
- TensorRT support matrix + Volta deprecation:
  <https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/support-matrix.html>,
  <https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.0.1.html>
- MuseTalk official bar (30fps @ 256px on a V100): <https://github.com/TMElyralab/MuseTalk>
- 5070 vs 5060 Ti 16GB: <https://www.tomshardware.com/pc-components/gpus/rtx-5070-vs-rtx-5060-ti-16gb>
- Super-refresh status: <https://wccftech.com/roundup/nvidia-rtx-5070-ti-super/>,
  <https://www.club386.com/nvidia-rtx-5070-super-release-date-specs-price-and-performance/>
- Used 3090 pricing: <https://resaleprices.com/gpu/nvidia-rtx-3090>
