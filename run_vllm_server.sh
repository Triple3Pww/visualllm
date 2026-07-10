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
# --- Model selector (2026-07-10): v2 (default) | v3 = Fun-CosyVoice3-0.5B --------------------
# COSYVOICE_MODEL=v3 switches to CosyVoice3: same 0.5B LM, a bigger 300M DiT flow decoder (v2's
# is 100M). Measured cost is ~+0.07s first-chunk TTFB under live MuseTalk render (_ab_run, 2 x 32
# samples) -- affordable. The engine's AutoModel dispatches on the yaml in the dir; v3 also
# REQUIRES its instruct prefix inside the reference transcript ('...<|endofprompt|>' SEPARATES the
# prefix from the transcript, llm.py:591 -- a bare/trailing marker yields empty/garbage audio).
# Explicit COSYVOICE_MODEL_DIR / COSYVOICE_PROMPT_TEXT still win; this only fills their defaults.
# NOTE: switching models needs THIS server relaunched in WSL -- the config panel's Restart cycles
# only the pipeline (:7860), never the WSL TTS server.
COSYVOICE_MODEL=${COSYVOICE_MODEL:-v2}
if [ "$COSYVOICE_MODEL" = "v3" ]; then
  export COSYVOICE_MODEL_DIR=${COSYVOICE_MODEL_DIR:-/mnt/e/Claude/cosyvoice-local-tts/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512}
  export COSYVOICE_PROMPT_TEXT=${COSYVOICE_PROMPT_TEXT:-"You are a helpful assistant.<|endofprompt|>你好，我是你的AI虚拟助手，很高兴见到你。今天天气不错，有什么我可以帮你的"}
  # Flow-decoder TensorRT for v3's 300M DiT. Verified 2026-07-10: the fp32 ONNX->TRT engine builds
  # in ~30s on first load (cached to flow.decoder.estimator.fp32.mygpu.plan), audio stays correct,
  # isolated first-chunk TTFB zh 1.48->1.08s / en 1.34->0.80s WITH CUDA graphs (EAGER=0 below) also on.
  # fp16 DiT TRT is NOT used (upstream warns of perf issues); fp32 only. Override with COSYVOICE_FLOW_TRT=0.
  # CAVEAT: graphs are the config the project rejected for v2 (P33: graph decode perturbs RAS -> zh
  # LIPSYNC degrades to the EYE though the TTS stopwatch looks faster). UNVERIFIED for v3's zh lipsync.
  export COSYVOICE_FLOW_TRT=${COSYVOICE_FLOW_TRT:-1}
fi
# Lever 2 (CUDA graphs): default 0 = capture graphs (faster per-token TTS decode).
# BASELINE 2026-07-10 (user decision): graphs ON, paired with v3 + flow-TRT, on LANGUAGE=en.
# Isolated first-chunk TTFB with graphs+flow-TRT: zh 1.08s / en 0.80s.
# KNOWN RISK, carried from the P27-P33 saga and NOT re-cleared for v3: graphs perturb the
# zh-critical RAS sampling (P18) -> the zh AUDIO gets longer + more internal silence, and since
# MuseTalk lip-syncs off a WHISPER of that waveform, the mouth can stop tracking the words. That
# is why graphs were kept OFF for the v2/zh baseline (the user's eye rejected them, P33). en is
# spared (no RAS reliance), which is why graphs-on is fine for the current en baseline. If you
# switch to zh, re-check the lipsync by eye (or flip COSYVOICE_VLLM_EAGER=1 / the panel toggle).
# (docs P27/P31/P32/P33; the config panel's CUDA-graphs toggle flips this + relaunches.)
export COSYVOICE_VLLM_EAGER=${COSYVOICE_VLLM_EAGER:-0}
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
