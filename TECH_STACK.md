# VisualLLm — Technology Stack

Real-time **speech → STT → LLM → TTS → photoreal talking-head avatar** system.
Multi-turn, streaming end-to-end. Goal: time-to-first-output **< 8 s**.

> Source of truth for current state is `STATUS.md`; this file is just the
> consolidated list of every technology in use.

## The active pipeline (one pure stack)

| Stage | Technology | Notes |
|-------|-----------|-------|
| **Orchestration** | [Pipecat](https://github.com/pipecat-ai/pipecat) `pipecat-ai` 1.3.0 | Real-time voice+video agent framework; wires every stage with streaming + barge-in |
| **Transport** | WebRTC (Pipecat WebRTC transport) | Browser ↔ server audio/video; serves the prebuilt client at `/client` |
| **VAD** | Silero VAD (local) | Voice-activity detection via `SileroVADAnalyzer` |
| **STT** | Deepgram nova-2 | `en-US` / `zh-TW` / `th` selected by `LANGUAGE` |
| **LLM** | OpenRouter | Any model via `OPENROUTER_MODEL` (default `google/gemini-2.5-flash-lite`); uses Pipecat's OpenAI-compatible `OpenAILLMService` |
| **TTS** | ElevenLabs `eleven_flash_v2_5` | Low-latency multilingual (also `multilingual_v2`) |
| **Avatar** | **Ditto** (antgroup/ditto-talkinghead, PyTorch path) | Local GPU lip-sync talking-head server; or `AVATAR=none` (client-rendered 3D face) |

## Avatar GPU server (Ditto)

| Component | Technology |
|-----------|-----------|
| Web server | **FastAPI** + **Uvicorn** (websocket streaming) |
| Inference | **PyTorch** 2.11 + CUDA 12.8 (cu128), on an RTX 5060 Ti |
| ONNX runtime | **onnxruntime-gpu** (needs CUDA-12 DLLs on path or falls back to CPU) |
| Face/landmarks | **MediaPipe** |
| Diffusion | LMDM (Ditto's motion diffusion model) |
| Misc | `filetype`, **NumPy** (replaces Ditto's Cython blend kernel — compiler-free) |
| Env | dedicated `ditto` conda env (Python 3.10), separate from the pipeline env |

Wire contract: client → server sends config/control JSON + 16 kHz mono PCM;
server → client returns 512×512×3 RGB frame buffers at a steady fps.

## Supporting libraries

| Library | Purpose |
|---------|---------|
| **python-dotenv** | `.env`-driven config |
| **loguru** | Logging (`[TTFO]` metrics, etc.) |
| **websockets** | Pipeline ↔ Ditto server client |
| **NumPy** | Audio/frame buffers |
| **asyncio** | Async pipeline runtime |

## Platform / runtime

- **Python 3.10** (pipeline env: pipecat; avatar env: conda `ditto`)
- **Windows 11** — no CUDA/C compiler on the box (NumPy + DLL-path workarounds in place)
- **Conda / Miniconda** for the GPU env
- **Browser client** = Pipecat prebuilt bundle (`pipecat-ai-prebuilt`), served as-is

## Configuration knobs (`.env`)

- `LANGUAGE` (`en` | `zh` | `th`)
- `TTFO_TARGET_SECONDS` (default 8)
- `AVATAR` (`ditto` | `none`), `CHARACTER_MODE` (`0` | `1`)
- API keys: `DEEPGRAM_API_KEY`, `OPENROUTER_API_KEY`, `ELEVENLABS_API_KEY` (+ `ELEVENLABS_VOICE_ID`)
- Ditto tuning: `DITTO_STEPS`, `DITTO_LEAD_FRAMES`, `DITTO_IDLE_GRACE`, `DITTO_SYNC_LEAD_S`, `DITTO_SYNC_FALLBACK_S`

## Removed / recoverable from git history

Earlier multi-provider sprawl, kept only in history: **MuseTalk**, **Simli**,
**HeyGen** avatars; **CosyVoice**, **F5-Thai** TTS; **FunASR**, **faster-whisper**
STT; **Azure** zh path; the echo-guard.
