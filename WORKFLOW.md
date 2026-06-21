# VisualLLm — Detailed Workflow

A real-time **speech → STT → LLM → TTS → photoreal talking-head avatar** system.
You talk, it transcribes you, an LLM answers, the answer is spoken, and a GPU renders a
lip-synced face — all streaming end-to-end over WebRTC to a browser. Target:
time-to-first-output (TTFO) **< 8 s**.

> This document is the *how it works* companion to `STATUS.md` (current state / decisions)
> and `CLAUDE.md` (repo conventions). When in doubt about live state, `STATUS.md` wins.

---

## 1. The big picture — two processes + a browser

The system is **two separate programs** plus the user's browser. They are intentionally
separate because the avatar render needs the GPU in its own environment.

| Process | Runs in | Port | Job |
|---|---|---|---|
| **Pipeline** (`pipeline/main.py`) | system Python 3.11 (has Pipecat) | **7860** | VAD→STT→LLM→TTS, serves the web client, owns the WebRTC connection |
| **Avatar server** (`local_services/ditto_server/app.py`) | `ditto` conda env (GPU, CUDA, TensorRT) | **8002** | Turns voice audio into 512×512 lip-synced RGB frames |

They talk to each other over a **local websocket** (`ws://localhost:8002/stream`). The
browser only ever talks to the **pipeline** (7860), over WebRTC.

```
  ┌─────────────┐        WebRTC         ┌──────────────────────────┐   local ws    ┌──────────────────┐
  │   BROWSER   │  mic up / A+V down    │   PIPELINE  (:7860)      │  16kHz PCM →  │  AVATAR (:8002)  │
  │ (your laptop)│ ◄──────────────────► │  STT→LLM→TTS→avatar glue │  ◄ RGB frames │  Ditto GPU render│
  └─────────────┘                       └──────────────────────────┘               └──────────────────┘
```

---

## 2. One turn, end to end

`pipeline/main.py` assembles a linear Pipecat `Pipeline`; frames stream through it. A
single conversational turn flows like this:

```
mic → transport.input() (+ Silero VAD)
    → STT          (Deepgram nova-2)         : audio → text  ("hello")
    → aggregator.user()                      : builds the user message
    → LLM          (OpenRouter)              : streamed, sentence-by-sentence answer
    → TTS          (Deepgram Aura / ElevenLabs): text → voice audio chunks
    → Avatar       (DittoVideoService)        : voice → lip-synced video (via :8002)
    → TtfoMeter                              : measures UserStoppedSpeaking → BotStartedSpeaking
    → transport.output()                     : audio + video → browser
;   aggregator.assistant()                   : records the bot turn into context
```

**It all streams.** The LLM's *first sentence* reaches TTS before the full answer is
generated, and TTS's *first audio chunk* reaches the avatar immediately. That overlap is
what keeps the whole thing inside the < 8 s budget.

`TtfoMeter` (`pipeline/metrics.py`) logs `[TTFO]` per turn — the gap from
`UserStoppedSpeakingFrame` to `BotStartedSpeakingFrame`.

---

## 3. Stage-by-stage detail

Each stage is built by a **thin, single-provider factory** in `pipeline/stages/`, driven
only by `.env` (no provider-selection branching — see `CLAUDE.md`).

### VAD — Silero (local)
`pipeline/stages/vad.py` (`build_vad_params`). Runs on the input audio to detect when you
**stop** speaking, which ends the user turn and kicks STT's final transcript.

### STT — Deepgram nova-2
`pipeline/stages/stt.py`. Streams your mic audio to Deepgram over a websocket; emits
interim + final transcripts. Language follows `LANGUAGE` (`en-US` / `zh-TW` / `th`). Needs
`DEEPGRAM_API_KEY`.

### LLM — OpenRouter
`pipeline/stages/llm.py`. Any model via `OPENROUTER_MODEL` (default Gemini Flash Lite). The
response is **streamed and sentence-aggregated** so TTS can start on sentence 1. A short
system prompt (in `pipeline/config.py`) keeps replies spoken-style (no markdown/emoji).
Needs `OPENROUTER_API_KEY`.

### TTS — Deepgram Aura (current) / ElevenLabs (default)
`pipeline/stages/tts.py`. Converts the LLM text to voice audio chunks, streamed.
- **Default**: ElevenLabs `flash_v2_5` (low latency, multilingual — covers zh-TW).
- **Fallback**: `TTS_PROVIDER=deepgram` → Deepgram **Aura** (reuses `DEEPGRAM_API_KEY`).
  English-only. Used when the ElevenLabs account is out of credits.
- Flip back by removing `TTS_PROVIDER` (or setting `elevenlabs`) and restarting.

### Avatar (client side) — `DittoVideoService`
`local_services/ditto_video.py`, a Pipecat `FrameProcessor` sitting between TTS and the
transport. It:
1. Resamples each TTS audio chunk to **16 kHz mono PCM** and sends it to the avatar server
   **immediately** (rendering can't wait).
2. **Buffers the downstream voice copy** and releases it frame-clocked to the real video
   the server reports rendering (so audio never runs ahead of the lips).
3. Pushes the returned video frames downstream, tagged for A/V sync (see §5).

### Avatar (server side) — Ditto on the GPU
`local_services/ditto_server/app.py` (FastAPI websocket). See §4.

### TtfoMeter
`pipeline/metrics.py`. Pure measurement — logs the < 8 s metric per turn and a summary on
disconnect.

---

## 4. The avatar subsystem in depth

### The wire contract (client ↔ server)
To understand either side you must read both (`ditto_video.py` + `ditto_server/app.py`):

**Client → server:**
- `{"type":"config","fps":12}` — sets output fps.
- `{"type":"speech_start"}` / `{"type":"speech_end"}` / `{"type":"reset"}` — turn markers / barge-in.
- binary **16 kHz mono PCM** chunks (the TTS audio).

**Server → client:**
- binary **512×512×3 RGB** frame buffers at a steady fps.
- `{"type":"video_start"}` / `{"type":"video_clock","frames":N}` / `{"type":"video_end"}` —
  sync markers counting only *real* rendered frames; these clock the client's voice release.

### Inside the server
- **Frame interception.** Ditto's `StreamSDK` normally writes frames to an mp4. We subclass
  it and override `_writer_worker` to divert finished frames onto a queue that a websocket
  `pump()` drains at a steady fps (showing an idle/neutral frame between turns).
- **Single-client + session lock.** The SDK is shared and single-client; `_session_lock`
  serializes whole sessions so a reconnect can't re-`setup()` over live worker threads.
- **Warmup masking.** The first diffusion has a ~2 s first-frame delay (FFD). The server
  captures a short **living idle loop** (blinks/micro-motion) once at startup and plays it
  during that gap instead of a frozen portrait (`DITTO_IDLE_CAPTURE_CHUNKS`).
- **Frame-drop to hit fps.** A `_StridedQueue` drops in-between motion frames *before* the
  costly render so a sub-realtime GPU keeps up at the chosen `DITTO_FPS`.

### TensorRT acceleration (`DITTO_TRT`, default on)
The GPU is Blackwell (sm_120). `build_trt.py` compiles FP16 TensorRT engines **for this
card** from Ditto's ONNX graphs (numerically validated vs fp32). `app.py::_enable_trt`
swaps **decoder / stitch / appearance / warp** to engines; diffusion + aux stay PyTorch.
The **warp** stage also runs as a TRT engine via a self-built GridSample3D plugin
(`grid_sample_3d_plugin.dll`). Net: the renderer reaches native ~25 fps. Engines are
GPU-arch-specific and cached in `checkpoints/ditto_trt_blackwell/`. `DITTO_TRT=0` = pure
PyTorch fallback.

---

## 5. A/V synchronization (how lips stay on the voice)

Sync is handled in **three places**; all must agree or audio/video drift.

1. **Frame-clocked voice release** (`ditto_video.py`). The voice is buffered and released
   paced to the server's `video_clock` markers (real rendered frames), so the voice waits
   when the render stalls and never drifts ahead. `DITTO_SYNC_LEAD_S` tunes a constant lip lead.

2. **`sync_with_audio` transport mode** (`DITTO_SYNC_WITH_AUDIO`, default on). Each turn's
   frames are tagged `OutputImageRawFrame.sync_with_audio=True` and routed through the
   transport's **audio queue**, so each frame displays at its audio position instead of on an
   independent video clock.
   - **Load-bearing coupling** (`main.py`): this only works when the transport is
     **non-live**. So `video_out_is_live = want_video and not DITTO_SYNC_WITH_AUDIO`. With
     `is_live=True`, tagged frames are silently dropped and video free-runs (drifts).

3. **One fps everywhere.** The server frame-drop stride, the client release clock, and
   `main.py video_out_framerate` must all equal **`DITTO_FPS`** (default 12).

The remaining felt delay at the start of a turn is Ditto's **diffusion warmup** (FFD,
~2 s) — masked, not removed (§4). Raising `DITTO_OVERLAP` shrinks it but destabilizes
lip-sync, so the validated sweet spot is `OVERLAP=25 / STEPS=25 / lead 0`.

---

## 6. Running the system (locally, on the GPU box)

Two terminals:

```bash
# 1. Avatar GPU server (its own conda env). Loads ~13GB of models — give it ~30-60s.
conda run -n ditto python -m local_services.ditto_server.app          # :8002
#    For live (unbuffered) logs, run the env python directly:
E:\miniconda3\envs\ditto\python.exe -u -m local_services.ditto_server.app

# 2. Pipeline (system Python). Serves /client.
python -m pipeline.main                                               # :7860 → /client
```

Then open `http://localhost:7860/client`, allow the mic, use **headphones** (no echo
guard), and talk. Fully close the tab between tries (sessions are serialized server-side).

**Sanity check without keys/network** (catches Pipecat import drift):
```bash
python -m scripts.preflight
```

---

## 7. Remote viewing (across the internet)

The avatar is WebRTC, so you view it natively in a remote browser — **never over RDP**
(RDP carries audio and video on separate paths and desyncs them; judge sync only natively).

The box runs **Tailscale Serve**, which proxies the pipeline over HTTPS on the tailnet:

```
https://porsche-pc.tail21bb8a.ts.net/client
```

- **HTTPS** is required so the browser allows the **microphone** (a plain `http://<ip>` LAN
  URL blocks the mic — insecure origin).
- Works from any device logged into the same Tailscale account.
- The pipeline binds to localhost only; Tailscale Serve proxies locally, so that's fine.

**Network reality:** over a cross-border link (e.g. Taiwan ↔ Thailand) there is real jitter.
Audio (~30 kbps) sails through; the video is heavier and can stutter or lag. Note the avatar
trailing-the-voice (vs random stutter) is the signature of **bandwidth starvation, not jitter** —
the encoder pushing more bits than the link reliably carries, so frames queue and fall behind. A
bigger jitter buffer can't fix that (it only adds lag); you must make the stream *fit the link*.
The Tailscale path is already direct WireGuard UDP (no DERP relay), so the wins are at the media
layer. Mitigations, in order of effectiveness for the photoreal path:
1. **Fit the stream to the link** (the real fix) — small frame (`DITTO_SIZE`, e.g. 320) + a
   bounded send bitrate (`WEBRTC_VIDEO_BITRATE_MAX`, ~600k) so the video can't starve the link;
   then a *small* jitter buffer (`CLIENT_JITTER_BUFFER_MS`, ~250) absorbs only the leftover
   timing variance. This is the Thailand→Taiwan recommended config.
2. **Receive-side jitter buffer** alone — `CLIENT_JITTER_BUFFER_MS` (default 400). Smooths jitter
   at the cost of latency; too high makes the avatar trail. After changing it, **hard-refresh**
   the browser (Ctrl+Shift+R) — the page is cached.
3. **Same LAN** (eliminates the WAN). Not always possible.
4. **Audio-only mode** (`AVATAR=none`) — face rendered on the client, only voice on the wire;
   immune to video jitter. Different (3D) face, not Ditto photoreal.
5. **Tier 2 (if 1 still strains)** — switch the codec to H.264 (better quality/bit, HW-decoded
   on the viewer) or move to the LiveKit transport (libwebrtc congestion control + FEC/NACK).

---

## 8. Configuration reference (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `LANGUAGE` | `en` | `en` / `zh` / `th` (STT + voice) |
| `AVATAR` | `ditto` | `ditto` (server photoreal) or `none` (client-rendered, audio-only) |
| `CHARACTER_MODE` | `0` | `1` = in-character Thai persona + emotion tags |
| `TTS_PROVIDER` | `elevenlabs` | `deepgram` = Aura fallback (reuses Deepgram key, en-only) |
| `DEEPGRAM_TTS_VOICE` | `aura-2-helena-en` | Aura voice when on Deepgram TTS |
| `OPENROUTER_MODEL` | `google/gemini-2.5-flash-lite` | any OpenRouter model |
| `CLIENT_JITTER_BUFFER_MS` | `400` | receive-side WebRTC jitter buffer (0 = off); ~250 for remote |
| `DITTO_SIZE` | `512` | square frame px (one value across server/client/transport); 320 for remote |
| `WEBRTC_VIDEO_BITRATE_MAX` | `600000` | VP8 send-bitrate ceiling, bits/s (0 = aiortc default 1.5M) |
| `WEBRTC_VIDEO_BITRATE` / `_MIN` | `500000` / `120000` | VP8 start-point / floor (graceful dip, no freeze) |
| `DITTO_FPS` | `12` | avatar output fps (one value across server/client/transport) |
| `DITTO_TRT` | `1` | TensorRT FP16 render path (0 = pure PyTorch) |
| `DITTO_OVERLAP` / `DITTO_STEPS` | `25` / `25` | diffusion window / sampling steps (sync sweet spot) |
| `DITTO_SYNC_LEAD_S` | `0` | constant lip-lead compensation |
| `TTFO_TARGET_SECONDS` | — | the < 8 s target for logging |

Keys required: `DEEPGRAM_API_KEY`, `OPENROUTER_API_KEY`, and (for default TTS)
`ELEVENLABS_API_KEY` + `ELEVENLABS_VOICE_ID`.

---

## 9. Key files

| File | Role |
|---|---|
| `pipeline/main.py` | Pipeline assembly, transport params, greeting, A/V-sync coupling, jitter-buffer injection |
| `pipeline/config.py` | All `.env`-driven config + system prompts |
| `pipeline/stages/*.py` | Per-stage single-provider factories (vad/stt/llm/tts/avatar) |
| `pipeline/metrics.py` | `TtfoMeter` (the < 8 s metric) |
| `local_services/ditto_video.py` | Client-side avatar processor + frame-clocked A/V sync |
| `local_services/ditto_server/app.py` | Ditto GPU server (ws, frame interception, pump, sync markers, TRT) |
| `scripts/preflight.py` | Import/drift check |
| `STATUS.md` | Live state + decision log (source of truth) |

---

## 10. Troubleshooting (failure modes we've actually hit)

| Symptom | Cause | Fix |
|---|---|---|
| Avatar shows but **won't talk** (voice + chat) | TTS provider out of credits/quota | Check provider account; swap `TTS_PROVIDER` or key |
| **Avatar not showing** at all | Ditto server (:8002) down | Start the avatar server; the pipeline needs it for `AVATAR=ditto` |
| Lips **drift / out of sync** in browser | `video_out_is_live=True` dropping synced frames | Keep `DITTO_SYNC_WITH_AUDIO=1` (couples is_live off) |
| Video **lags / stutters** remotely, audio fine | WAN jitter on the heavier video stream | Tune `CLIENT_JITTER_BUFFER_MS`; or lower `DITTO_FPS`/size; or audio-only mode |
| **Avatar trails the voice** | jitter buffer too high (video delayed) | Lower `CLIENT_JITTER_BUFFER_MS` |
| ~2 s pause before lips move each turn | Ditto diffusion **warmup (FFD)** — fundamental | Masked by the idle loop; raise `DITTO_OVERLAP` only with care (breaks sync) |
| Avatar **laggy on the GPU box itself** | onnxruntime fell back to CPU, or fps mismatch | Verify CUDA DLLs on path; keep one `DITTO_FPS` everywhere |
| Judging sync **over RDP** looks wrong | RDP desyncs audio/video paths | Judge natively (remote browser) or via `capture_mp4.py` offline |
