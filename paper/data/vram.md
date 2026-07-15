# VRAM budget (measured 2026-07-16, live stack, between measurement turns)

Method: Windows WDDM hides per-process GPU memory from `nvidia-smi` (`[N/A]`), so per-process
numbers come from the `\GPU Process Memory(*)\Dedicated Usage` performance counters; the
whole-card number from `nvidia-smi --query-gpu=memory.used`.

| process | role | dedicated GPU memory |
|---|---|---|
| `vmwp` (WSL2 VM) | CosyVoice2-0.5B TTS on vLLM (KV pool at 7% util, max_len 2048) | 2,312 MiB |
| `python` (musetalk env) | MuseTalk render server (TensorRT engines, torch UNet/VAE freed) | 3,286 MiB |
| `python` (pipeline) | Pipecat pipeline + Silero VAD (CPU inference) | < 100 MiB (below counter threshold) |
| **project total** | | **≈ 5.6 GB** |
| whole card (incl. desktop apps) | RTX 5060 Ti 16 GB | 6,451 / 16,311 MiB |

Notes:
- Matches the session-24 per-process record (TTS ~2.3 GB, avatar ~3.3 GB). The session-24
  whole-card figure (~7.8 GB) included a heavier desktop; the project share is what the paper
  should cite.
- Idle-adjacent measurement: taken between turns of the zh campaign, i.e. models loaded and
  warm. Peak during a turn adds transient activations but the vLLM pool and TRT workspaces are
  preallocated, so the resident numbers above are the stable footprint.
