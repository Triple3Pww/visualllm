# local_services

The local pieces of the pipeline: the Pipecat client wrappers, the GPU model servers they
talk to, and the two custom browser clients.

## Pipecat client wrappers

Dropped into the pipeline by the factories in `pipeline/stages/*.py`, chosen by `.env`.

| Module | Class | Stage |
|--------|-------|-------|
| `cosyvoice_tts.py` | `CosyVoiceTTSService` | streaming TTS (HTTP client). Also serves `TTS_PROVIDER=jaitts` — same `/tts/stream` raw-PCM contract, just a different URL |
| `sherpa_stt.py` | `SherpaStreamingSTTService` | offline streaming STT (`STT_PROVIDER=sherpa`), in-process on CPU, ~0 VRAM |
| `musetalk_video.py` | `MuseTalkVideoService` | the lip-sync avatar (websocket client) — **owns the A/V sync**; read it before touching sync |
| `first_piece_aggregator.py` | `FirstClauseAggregator` | the TTFO first-clause split + the filler-word opener |
| `weather_chain_llm.py` | `WeatherChainLLMService` | `LLM_PROVIDER=weather_chain` — the NCU Chinese weather bot |
| `avatar_memory.py` | `MemoryStore` | the weather bot's growing memory (local CPU qwen). Only active with `weather_chain` |

## Servers

| Folder | Port | Env | Talks to |
|--------|------|-----|----------|
| `musetalk_server/` | 8002 | `musetalk` conda env | `MuseTalkVideoService` |
| `jaitts_server/` | 8004 | the shared F5 venv | `CosyVoiceTTSService` (Thai only) |
| `config_panel/` | 7870 | system python | the browser — edits `.env` + restarts the stack |

**The CosyVoice TTS server is NOT here** — it lives at **`tts/cosyvoice-server/`** (repo root) and runs
on vLLM inside WSL on `:8001`. It was merged into this repo on 2026-07-14; before that it was a separate
repo, and a stale copy of it here had silently rotted.

`musetalk_server/` defaults to the **TensorRT** render path (`MUSETALK_TRT=1`, ~1.5× faster — it is what
holds A/V sync under shared-GPU contention; `docs/PROBLEMS-AND-FIXES.md` P16). Engines (`trt_cache/`,
~1.75 GB, gitignored, GPU-specific) are built once with `musetalk_server/trt_build.py`; any load failure
falls back to PyTorch silently.

## Browser clients

`studio_client/` (`/studio/`) — a self-contained vanilla-JS page, no build step, mounted by
`pipeline/main.py`. It speaks the same `POST /api/offer` SmallWebRTC signaling as the prebuilt `/client`
bundle, and additionally does the split-mode mouth-crop compositing, the chat transcript, and (since
2026-07-14) the receive-side jitter buffer + the phone-loudspeaker route — all fed by
`GET /client/ice-config`. (`nimbus_client/` was a second, ~740-line verbatim JS copy of this page
differing only in theme; removed 2026-07-14 — `/studio/` is now the single custom client for every
avatar preset, including "nimbus".)

**`/client` (the pipecat prebuilt bundle) is unsupported while `MUSETALK_SPLIT=1`** — it cannot composite,
so it would show a floating mouth crop. Use `/studio/`.

## Removed 2026-07-14

`moss_server/` (MOSS-TTS-Realtime, `:8003`), `funasr_server/` + `funasr_stt.py` (SenseVoice STT). Neither
was ever selected in `.env` — an untried fallback is not a safety net. Both are in git history.

## VRAM reality (16 GB, shared)

CosyVoice on vLLM (~4 GB) and MuseTalk (~5 GB) share the one card. **Load order matters: start CosyVoice
BEFORE MuseTalk** — vLLM needs the card mostly free to claim its KV cache, or it dies with "No available
memory for the cache blocks" (`docs/PROBLEMS-AND-FIXES.md` P15). `scripts/launch.ps1` already does this.
