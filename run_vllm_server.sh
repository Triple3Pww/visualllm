#!/usr/bin/env bash
# Launch CosyVoice2 TTS on vLLM, in WSL (Ubuntu) on the Blackwell 5060 Ti.
# This replaces the Windows PyTorch CosyVoice server on :8001 and cuts first-chunk
# latency ~3.4s -> ~1.2s (vLLM accelerates the autoregressive speech-token LLM).
#
# Why each env var (all required on this bleeding-edge stack -- see the build notes):
#   COSYVOICE_VLLM=1                  -> engine loads the LLM on vLLM (tts_engine.py switch)
#   VLLM_ENABLE_V1_MULTIPROCESSING=0 -> run the engine in-process (no spawn re-import crash)
#   VLLM_USE_FLASHINFER_SAMPLER=0    -> native torch sampler (flashinfer's needs nvcc, not present)
#   CC/CXX + PATH                    -> Triton JITs kernels at runtime; point it at the conda gcc
#   COSYVOICE_VLLM_EAGER=1 (default) -> skip torch.compile/CUDA-graph capture (needs more toolchain)
set -e
ENV=/home/porsche/miniconda3/envs/cosyvllm
export PATH=$ENV/bin:$PATH
export CC=$ENV/bin/gcc
export CXX=$ENV/bin/g++
export COSYVOICE_VLLM=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
# Lever 2 (CUDA graphs): EAGER by default (0 = capture graphs -> faster per-token TTS decode).
# VERDICT 2026-07-05 (8th session): KEEP EAGER. The graph win is real but ONLY on the TTS side; the
# COST is on the zh-audio/avatar side, which is what matters for the talking head.
#   - A TTS-TTFB variance probe (_ttfb_variance.py) showed graphs FASTER + LOWER-variance than eager
#     (even under real MuseTalk render) -> so from the TTS stopwatch, graphs look strictly better.
#   - BUT the user's eye caught zh LIPSYNC degrading with graphs ON. Root cause, measured
#     (_zh_audio_ab.py, same zh sentence x5): graphs ON alters the zh AUDIO -- longer (median
#     8.92 vs 8.28s), MORE internal silence (worst 0.76 vs 0.68s, frac 0.30-0.36 vs 0.22-0.33),
#     more run-variance. The graph decode perturbs the zh-critical RAS sampling (the P18 fix that
#     stops zh looping on the silence token). MuseTalk lip-syncs off a WHISPER of that waveform, so
#     a degraded zh waveform -> mouth shapes that don't track the words. en is spared (no RAS reliance).
# So the "no drawback" re-investigation measured the WRONG side (TTS TTFB); P31's original revert was
# right. Set COSYVOICE_VLLM_EAGER=0 to force graphs (fine for an en-only / TTS-throughput setup).
# (docs P27/P31/P32/P33; the config panel's CUDA-graphs toggle flips this + relaunches.)
export COSYVOICE_VLLM_EAGER=${COSYVOICE_VLLM_EAGER:-1}
# zh TTFO lever -- REVERTED to 0 (2026-07-04, live-measured twice): hop=5's isolated first-chunk
# TTFB win (~2.5s -> ~1.8s) is ERASED live in steady mode -- the SMALLER opening chunk fills the
# MUSETALK_LEAD_FRAMES cushion slower, so the synced voice-start is DELAYED (P19 grid + a fresh
# 2026-07-04 A/B: live zh TTFO median ~4.1s @hop=5 vs ~3.1s @hop=0, smoothness screen clean,
# zh steady-hold 1.9-2.2s -> ~0.8s). hop>0 remains only a knob for experiments.
# (Per-language plumbing kept: tts_engine.py::_apply_first_hop, per request, zh-only by is_cjk.)
export COSYVOICE_FIRST_HOP_ZH=${COSYVOICE_FIRST_HOP_ZH:-0}
# VRAM trim (2026-06-30, measured): cap max sequence length + the card fraction vLLM may use.
# CosyVoice generates ONE short sentence of speech tokens per request, so the default KV
# reservation (max_model_len 32768) is wildly oversized. Capping max-len to 2048 lets the
# util fraction drop far below the old ~0.25 "floor": at 0.16 -> vLLM ~3.7GB with 74x KV
# headroom; at 0.12 -> ~3.4GB / 47x (verified: en + a 27s zh paragraph synth clean, no
# truncation). 0.16 is the robust default; set COSYVOICE_VLLM_GPU_UTIL=0.12 to squeeze the
# whole stack under 8GB. Lower util also reserves less of the shared card -> friendlier to
# the MuseTalk load-order (less "No available memory for the cache blocks"). Override either.
export COSYVOICE_VLLM_MAX_LEN=${COSYVOICE_VLLM_MAX_LEN:-2048}
export COSYVOICE_VLLM_GPU_UTIL=${COSYVOICE_VLLM_GPU_UTIL:-0.16}
cd /mnt/e/Claude/cosyvoice-local-tts
exec $ENV/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8001
