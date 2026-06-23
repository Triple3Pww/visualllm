# Design: Weather-chain LLM node

_2026-06-23 — baseline `cd88f20`. Status: approved (scope + approach), pending spec review._

## Goal

Replace the general OpenRouter LLM at the pipeline's LLM slot with a **dedicated Chinese
weather assistant** that calls a collaborator's LangServe weather-chain endpoint. Every user
turn becomes a weather question routed to the chain; the chain's streamed answer flows through
the unchanged TTS → MuseTalk avatar path. The avatar becomes a spoken Mandarin weather bot.

The endpoint (NCU, Taiwan):

```
POST http://140.115.54.87:8000/chain/resWeatherChain/stream
Content-Type: application/json
{"input": {"query": "明天台北市有下雨嗎?", "model": "gemma3:27b"}}
```

It is a LangServe `add_routes` Runnable `/stream` route → Server-Sent Events (SSE). The chain
**requires the query in Chinese** and answers in Chinese.

## Why this is a clean swap

In pipecat 1.3.0 the LLM slot only owes the rest of the pipeline one contract (verified in
`pipecat/services/openai/base_llm.py::process_frame`): on each inbound `LLMContextFrame`, emit

```
LLMFullResponseStartFrame → LLMTextFrame(chunk)* → LLMFullResponseEndFrame
```

TTS sentence-aggregation, the avatar, and `aggregator.assistant()` all key off those frames and
do not care about the source. So swapping the brain touches **only** the LLM stage — nothing
downstream changes.

## Approach (chosen: A — custom `LLMService` subclass)

A new `WeatherChainLLMService(LLMService)` that speaks pipecat on the outside and the weather
chain on the inside. Rejected alternatives: **B** (bare `FrameProcessor` on raw transcripts —
bypasses the aggregator's turn/echo-guard handling and forces hand-rolled start/end markers, a
fragile foot-gun) and **C** (a local OpenAI-compatible shim process — a 4th server to babysit and
*two* format translations instead of one; the proxy pattern only earns its keep across many tools).

## Components

### 1. `local_services/weather_chain_llm.py` — `WeatherChainLLMService(LLMService)`

New file (services live in `local_services/`, like `cosyvoice_tts.py`). Subclasses pipecat's
`pipecat.services.llm_service.LLMService` so it inherits the conveyor-belt etiquette
(metrics, start/stop, error plumbing).

- **`process_frame`** — mirror `BaseOpenAILLMService.process_frame`: on `LLMContextFrame`, push
  `LLMFullResponseStartFrame()`, `start_processing_metrics()`, call `_process_context(context)`,
  and in `finally` push `LLMFullResponseEndFrame()` + `stop_processing_metrics()`. Pass every other
  frame through unchanged.
- **Query extraction** — from `context.get_messages()`, take the **last `role == "user"`
  message**'s text as `query`. The chain is stateless and weather-only — no history is sent (the
  system prompt and prior turns are irrelevant to it).
- **The call** — `httpx.AsyncClient` streaming POST to `{WEATHER_CHAIN_URL}/stream` with
  `{"input": {"query": query, "model": WEATHER_CHAIN_MODEL}}`. Reuse one client instance across
  turns (open it lazily; close on `stop`/`cancel`).
- **SSE parse** — read line-by-line. LangServe `/stream` emits `event: <name>` / `data: <json>`
  pairs. For `event: data`, JSON-decode the `data:` payload and extract text **tolerantly**: the
  chunk may be a bare JSON string (`"明天"`), or an object with a `content`/`output` field
  (depends on the chain's output parser). Push each non-empty piece as `LLMTextFrame`. Stop on
  `event: end` (or stream close).
- **Errors** — on connection failure / HTTP error / timeout: `push_error(...)` and emit one short
  spoken Chinese fallback (e.g. "抱歉，天氣服務暫時連線不上。") as an `LLMTextFrame` so the avatar
  isn't silently dead. Never raise out of `process_frame` (the base wraps it, but keep the spoken
  fallback explicit).

### 2. `pipeline/config.py` — provider switch + chain knobs

Keep OpenRouter as a deliberate **fallback switch** (the repo's convention, not multi-provider
branching), exactly like `TTS_PROVIDER`:

- `llm_provider: str = _get("LLM_PROVIDER", "openrouter")` → `openrouter` | `weather_chain`.
- `weather_chain_url: str = _get("WEATHER_CHAIN_URL", "http://140.115.54.87:8000/chain/resWeatherChain")`
  — base; the service appends `/stream`.
- `weather_chain_model: str = _get("WEATHER_CHAIN_MODEL", "gemma3:27b")`.

### 3. `pipeline/stages/llm.py` — factory branches on provider

`build_llm(cfg)` returns `WeatherChainLLMService(...)` when `cfg.llm_provider == "weather_chain"`,
else the existing `OpenAILLMService`. Thin, single-provider per branch — matches the other
factories. Import the weather service lazily inside the branch (keep preflight import-clean).

### 4. `pipeline/main.py` — warmup guard + greeting

- `_warmup_llm()` currently calls `llm._client.chat.completions.create(...)` — OpenAI-only; it
  would crash on the weather service. Guard it: only run the chat-completions warmup when the
  service exposes that client (e.g. `hasattr(llm, "_client")` / provider check); otherwise no-op
  (the chain has no cheap warmup ping).
- The startup log line and the greeting: when `weather_chain`, log the weather provider and use a
  Chinese weather-themed greeting (the existing `is_mandarin` greeting already covers Chinese;
  optionally tailor it to weather).

### 5. `.env` for the demo + `scripts/` probe

- `.env`: `LLM_PROVIDER=weather_chain`, `LANGUAGE=zh` (Deepgram zh-TW in, CosyVoice female-Mandarin
  out — already the natural fit). Document the new knobs in `.env` / WORKFLOW.md §8.
- `scripts/probe_weather_chain.py` — a tiny standalone POST that prints the raw SSE bytes, so the
  exact chunk shape can be confirmed the instant the server is reachable and the parser tweaked in
  one line if LangServe's framing differs from the assumed `event: data` / `data: <json-string>`.

## Data flow (one turn)

```
mic → STT(zh) → aggregator.user() → LLMContextFrame
   → WeatherChainLLMService:  last user msg → POST .../stream
                              ← SSE tokens → LLMTextFrame*
   → TTS(CosyVoice) → MuseTalk avatar → browser
   aggregator.assistant() records the bot turn
```

## Risks / flags (not hidden)

- **Server unreachable now.** Port 8000 refuses connection (host pings, 78 ms). Built to the
  standard LangServe `/stream` contract; the probe + a possible 1-line SSE-parse tweak close the
  gap once it's up. **The SSE chunk shape is an assumption until verified.**
- **TTFO < 8 s may not hold.** `gemma3:27b` on Ollama in Taiwan + a likely weather-retrieval step
  may be slower than OpenRouter. Streaming first-token mitigates; measure with `scripts/measure`.
- **No conversation memory** by design — each turn is an independent weather query. Acceptable for
  the demo; revisit only if follow-up questions ("那後天呢?") need context.

## Out of scope

Intent routing (weather vs general), tool-calling, multi-provider blending, and any change to STT/
TTS/avatar/sync. Reverting to general chat is a one-line `.env` flip (`LLM_PROVIDER=openrouter`).

## Testing / verification

- `python -m scripts.preflight` — imports resolve with no keys/network.
- `python scripts/probe_weather_chain.py` — confirm live SSE shape (when reachable).
- Live: `LANGUAGE=zh LLM_PROVIDER=weather_chain`, open `/client/`, ask a Chinese weather question,
  confirm the avatar speaks the answer; check `[TTFO]` in `pipeline.log` and `scripts/measure`.
