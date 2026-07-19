# VisualLLm — Why the Code Is This Way (current state)

A **current-only** reference: for each live setting or hard-won decision, the **symptom** you'd
observe, the **root cause** (as proven, not guessed), the **fix that is live now**, and the
**why**. Organised by subsystem, in pipeline order. Newer than any single P-entry, because it
states each truth *once* at its final resolution — the reversals, dead-ends, and superseded
mechanisms are deliberately dropped.

For the full debugging archaeology of any item, follow its `(full story: P##)` pointer into
**`PROBLEMS-AND-FIXES.md`** (and `PROBLEMS-AND-FIXES-ARCHIVE.md` for retired numbers). Those two
files are the lossless history; this file is the clean map.

> Companion docs: `STATUS.md` (source of truth for live state) · `WORKFLOW.md` (end-to-end
> workflow + full `.env`) · `CLAUDE.md` (conventions). Live measurement: `python -m scripts.measure`.

---

## 0. How we debug here (the method, before any fix)

These rules are timeless — every audio/sync bug below was found by obeying them and prolonged by
breaking them.

- **Judge audio from a CONCATENATED WAV, never per-chunk RMS.** aiohttp `iter_chunked` / pipecat
  frames are **not sample-aligned** — one chunk can split mid-int16, so a single chunk reads as
  "loud garbage" while the concatenated stream is clean. Window ≥0.5 s, skip silence (`rms<0.005`),
  use spectral flatness (noise ≈ 0.5+, speech ≈ 0.0–0.05). This cost multiple hours more than once.
- **Your reference must not share the suspect input.** "Delivered frames == offline render,
  byte-identical" only proves the render is *deterministic* — feed both sides the same corrupt PCM
  and it passes. An offline render fed a voice captured off **WebRTC** gets the *repaired* downstream
  copy and always looks good. Capture the true delivered bytes: `MUSETALK_DUMP_PCM=1` +
  `MUSETALK_DUMP_DELIVERED=1`.
- **Mouth-motion vs audio-RMS correlation is useless** (it has misled 4×). Never verify A/V sync
  from a WebRTC capture reconstructed by *arrival* time — under `steady` the voice is released in
  bursts paced to the render.
- **Bisect at component boundaries with WAV captures**: source (CosyVoice) → avatar-out →
  transport-in → transport-out. Whichever boundary flips clean→garbled is the culprit.
- **The GPU is shared** (CosyVoice + MuseTalk on one card). Contention is real but GPU math does not
  *corrupt* under load, it only *slows*. Garbage bytes ⇒ a logic/library bug, not contention.
- **Reproduce headlessly**: `python -m scripts._webrtc_probe --mic output/_zh_q_def.wav --lead 8`
  drives a real turn with no browser; `python -m scripts.measure --offline-capture` wraps it.

---

## 1. Audio integrity — PCM sample alignment

**Symptom.** "Live lipsync is bad": the mouth flaps in a generic wordless pattern that never closes
for pauses, while the *voice itself sounds perfect*. The multi-session trap.

**Root cause.** Audio is int16 (2 bytes/sample); the CosyVoice server writes whole samples, but
HTTP-chunk/TCP boundaries land mid-sample and `iter_chunked()` propagates them (several odd-length
chunks per utterance, measured). Drop or mis-pair one byte and every following int16 is assembled
from the wrong two bytes → **loud broadband noise**. MuseTalk lip-syncs off a **Whisper of the
waveform**, so the noise makes the mouth move in a generic pattern — but the delivered voice (a
separate copy) still sounds fine, which is why it hid.

**Fix (live).** Alignment is enforced at the **producer**, `cosyvoice_tts.py::run_tts`: it carries
the dangling byte across reads so every `TTSAudioRawFrame` it yields is whole-sample. This is the
one place the invariant is restored.

**Why here and nowhere else.** If odd buffers ever reappear, fix the producer — **never drop a byte
downstream**. The old consumer-side patches (`_align_even`, `_srv_carry`) were removed; a
downstream drop is exactly what caused the noise. `(full story: P52; class also covers P3/P34/P40.)`

---

## 2. A/V sync

### 2.1 `steady` is the default sync mode
**What it is.** `MUSETALK_SYNC_MODE=steady` (video-master): the voice is buffered and released
**paced to the real frames the server reports rendering**, so it never drifts ahead and the turn
starts synced (the user's pick). `live` (audio-master) forwards the voice immediately and lets the
lips trail ~0.75 s — the robust alternative that never pauses.

**The steady tradeoff.** Under a long render stall the voice briefly **pauses**, then resumes clean.
Switch to `live` only if that pause is worse than the lip trail. (Under `steady`, "lips drift behind
the voice" is *not* the failure mode — an underflow is a voice *pause*, not drift.)

### 2.2 The `is_live` coupling (critical, load-bearing)
**Root cause of a whole class of silent desync.** Pipecat 1.3.0 only reads `_video_images` (where
per-frame audio-pinned frames land) when `video_out_is_live=False`; with `is_live=True` the tagged
frames are silently dropped and video free-runs. So in `main.py`:
`video_out_is_live = not config.avatar_sync_with_audio` — **never set `is_live` independently.**
`(full story: P-sync in the archive / A/V-sync memory.)`

### 2.3 One fps everywhere
The server frame-drop stride, the client release clock, and `main.py video_out_framerate` must all
equal `config.avatar_fps` (`MUSETALK_FPS=12`) or audio/video drift. The server now recomputes its
frame interval every tick from the client's `config` message (not once from its own env), so a panel
edit that restarts only the pipeline can't leave `:8002` on a stale fps — **but keep the values
equal anyway**; that closed the silent-failure mode, it didn't make a mismatch correct. `(P47.3.)`

### 2.4 Per-frame audio coupling (proto 2)
**Root cause it replaced.** The client used to pair voice to frames by `i/fps` index arithmetic and
*guess* which frames were held re-sends via byte-compare — an fps mismatch shifted the whole
audio↔lip mapping.

**Fix (live).** With `"proto": 2` the server prefixes every binary frame with a 16-byte header
(`MTF2` | kind u8 | audio_pos u64): kind 0 = real render / 1 = held / 2 = idle-neutral, and
`audio_pos` = cumulative real 16 k samples the frame covers. The steady client releases voice paired
to the server's own `audio_pos`, and held frames are *declared*, not guessed. The header carries a
*position*, not audio bytes — the delivered voice stays the original 24 kHz TTS audio. Offline
harnesses that never ask for proto 2 keep the bare-frame wire byte-identical. `(full story: P51.)`

### 2.5 `MUSETALK_LEAD_FRAMES=14` — CLOSED at 14
**Why not lower.** 14 is the synced-start cushion *and* a mid-turn shock absorber; lower starves the
render queue → the avatar freezes. The user live-eye tested **every value below 14** (incl. `lead=8`,
which measured a clean zh 3.03 / en 2.48 s on the probe) and saw delay or freezes. The probe misses
what the eye catches. **Do not re-try lower leads.** This knob reaches the avatar server only via a
full relaunch (`launch.ps1`/`run.ps1`) — the config panel's Restart cycles the pipeline only.
`(full story: P19/P22.)`

### 2.6 Turn-start feed burst
`MUSETALK_FEED_BURST_S=1.0` — the client bursts the first 1 s of each turn's audio to the server
un-paced (the rest is real-time-paced) so the renderer isn't starved at turn start. Cut lip-start
lag ~1.9 s → ~0.8 s.

### 2.7 Frame count = audio length (no early finish)
**Symptom (fixed).** On long replies the lips finished ~1–2 s before the voice. **Root cause:** the
old `int(16000/fps)` per-segment sizing truncated ~1 frame/segment. **Fix:** the server sizes
segments with a `samples_for_frames` **ceil**, so rendered frames = audio_sec × fps (±1) even for a
non-divisor fps like 14. The end-of-turn leftover-audio blip is fixed at the root by proto 2 (each
frame pairs to its own `audio_pos`); **do not re-add the old `audio_cap` ceil patch**. `(full story:
P9, P51.)`

### 2.8 End-of-turn close (no snap)
`MUSETALK_END_TAIL_FRAMES=0` + `MUSETALK_CLOSE_FADE_FRAMES=5`: the client cross-dissolves the last
**spoken** frame → rest pose over 5 frames (~0.42 s @12fps), delivered **free-run/untagged** (a
synthesized blend has no `audio_pos` to pair to and the voice is already drained, so a tagged frame
would never clock out of the non-live transport). `END_TAIL=0` is required — a neutral tail would
make the crossfade start from neutral (a no-op). Measured on the delivery path 2026-07-17: `fade=0`
puts 77 % of the close in one frame, `fade=5` puts 16 %. `(full story: P12.)`

### 2.9 The steady "screech" is fixed
`steady` used to intermittently screech mid-reply. **Root cause:** after a >3 s render-stall gap
pipecat fired `_bot_stopped_speaking()` and discarded the partial (odd-length) audio buffer →
the rest of the turn's PCM went odd-misaligned → broadband noise (the same class as §1). **Fix:**
`main.py::_relax_bot_vad_stop_timeout()` raises `BOT_VAD_STOP_FALLBACK_SECS` (we drive an explicit
`TTSStoppedFrame` per turn anyway) + the producer-side alignment in §1. A stall now just pauses the
voice and resumes clean. `(full story: P3/P52.)`

---

## 3. Avatar render & GPU

### 3.1 TensorRT render is the baseline
`MUSETALK_TRT=1` — TensorRT UNet+VAE engines. At the live config (SIZE=512 / SPLIT=1 / fps=12)
render is 455 → 171 ms per 8-frame segment, lifting headroom 1.5× → 3.9× clean and 1.04× → 2.05×
under GPU contention. **What it buys today is MARGIN, not a visible fps=12 fix** — at 12 fps PyTorch
also clears the 667 ms budget (by 4 %), so flipping `TRT=0` shows the same flat drift. To *see* the
difference, tighten the budget (`_drive_frames … 25`): PyTorch collapses to +4.04 s on a 13.6 s
reply while TRT holds +0.32 s flat. Engines live in `musetalk_server/trt_cache/` (~1.75 GB,
gitignored, GPU/driver-specific — rebuild with `trt_build.py`); any load failure silently falls back
to PyTorch. `(full story: P16.)`

### 3.2 GPU composite (opt-in follow-on)
`MUSETALK_GPU_COMPOSITE=1` — runs the per-frame mask-blend + downscale on the GPU (torch) instead of
CPU PIL/cv2: composite ~73 → ~11 ms/segment, total render 246 → 182 ms (−26 %). Only active with
`MUSETALK_TRT=1` (the VAE output is already a GPU tensor there). Output is pixel-identical (SSIM 1.0,
≤1 LSB); falls back to CPU if a crop_box runs off-frame. At 12 fps it does **not** reduce A/V drift
(TRT already holds ≥12 fps) — the win is reserve headroom + a freed CPU. `(full story: P17.)`

### 3.3 `cudnn.benchmark` MUST stay `False`
**Root cause.** With it `True`, cuDNN re-autotunes on the turn-START segment (a different shape than
mid-turn) → a **~16 s GPU spike on the first segment of every turn** → lips start ~5 s late and the
render falls behind on long replies. `False` removed it with no change to steady-state per-frame
time. Diagnose render timing with `MUSETALK_PROFILE=1`. `(full story: P1.)`

### 3.4 The 512px sync ceiling
`MUSETALK_SIZE=512` (+ `MUSETALK_BASE_MAX=768`) is the crispness ceiling that still holds ≥fps under
**real** shared-GPU load. 768/1024 profile fine in isolation but drop to ~10 fps under live CosyVoice
contention → steady-mode voice lag. Higher res needs a dedicated avatar GPU. The animated mouth stays
MuseTalk's fixed 256 px regardless — SIZE only sharpens the static frame/background. Keep `MUSETALK_FPS`
identical across server + pipeline. `(full story: P36.)`

### 3.5 VRAM budget on the shared 16 GB card
vLLM and MuseTalk share one card. Live settings that keep ~9.5 GB free with all three up:
- `COSYVOICE_VLLM_GPU_UTIL=0.07` (set in `run_vllm_server.sh`, not `.env`) — with `MAX_LEN` capped at
  2048 the KV need is tiny, so 0.07 (≈35× a real turn) is verified clean. On vLLM 0.23.0 the budget is
  `card × util` and other processes are **not** charged against it; the wall is ~util 0.79.
- `MUSETALK_FREE_TORCH=1` — once the TRT engines load, the torch UNet+VAE are dropped (−1.8 GB), since
  the fallback decision already happened at load time. (Gotcha: null the **inner** attrs `unet.model`
  / `vae.vae`, not just `self.unet`, or nothing frees.)

**Silent bot ≠ load-order bug.** If the avatar shows but is silent, first check `:8001` is up
("Cannot connect to host …:8001" / "Available KV cache memory" must be positive) — load order is kept
as free insurance, not a rule (vLLM starts fine second onto an occupied card at util 0.07). `(full
story: P15/P50.)`

---

## 4. TTS latency — CosyVoice

### 4.1 vLLM in WSL, reached by the WSL IP
CosyVoice's autoregressive LLM runs on **vLLM inside WSL Ubuntu** (`cosyvllm` env), cutting first-chunk
TTFB ~3.4 s → ~1.1 s. The pipeline reaches it via `COSYVOICE_URL` set to the **WSL IP, NOT
`localhost`** — WSL2's localhost relay buffers the streaming audio ~2 s. The IP changes on
`wsl --shutdown`, so `launch.ps1` auto-heals a stale `.env` value against `wsl hostname -I` on every
start (`Sync-CosyVoiceUrl`); only manual non-launcher starts need a hand-update. `(full story: P6.)`

### 4.2 CUDA graphs are ON for every language, zh included
`COSYVOICE_VLLM_EAGER=0` (graphs on) is correct for **every** language. Graphs are the TTS-first-chunk
win (avg ~2.0 → ~0.85 s) and lower-variance than eager even under real render. Graphs slightly alter
the zh waveform, but the user's live eye confirms zh lipsync is fine — **a measurable delta is not a
perceived one; the live eye is the arbiter.** Do not flip it back to eager. `(full story: P27; the
P33 "graphs degrade zh lipsync" verdict is reversed.)`

### 4.3 First-chunk TTFB scales with input length — split the opener
CosyVoice prefills the whole sentence before the first audio token, so TTFB tracks **length, not
language** (≈ 0.648 s + 25.9 ms/char). Two levers, because the splitters differ:
- **English:** `COSYVOICE_FIRST_PIECE=1` (min/max 18/32 chars) emits a short opening *clause* first —
  splits at an ASCII comma/space past the thresholds.
- **Chinese:** `COSYVOICE_FIRST_PIECE_ZH=1` (min 5 CJK chars) flushes the first piece at a full-width
  **，；：ONLY, never a char cap** (a cap cuts mid-word 天氣預|報; a comma boundary can't). The English
  split never fires on zh (ASCII comma/space vs full-width, no spaces), which is why zh needs its own.

Long-opener turns dropped ~4.78 → ~3.08 s with no between-clause pause. Note `COSYVOICE_FIRST_HOP_ZH=0`
(a smaller opening chunk *hurt* live zh — it filled the lead cushion slowly). `(full story: P19/P22/P23/P56.)`

### 4.4 Chinese silence-loop fixed (RAS restored)
**Symptom (fixed).** zh intermittently looped on the silence token → a ~4 s sentence became ~12 s of
dead silence, heard as "halting" speech while the avatar kept moving. **Root cause:** running the LLM
on vLLM dropped CosyVoice's repetition-aware sampling (RAS). **Fix:** RAS restored as a vLLM logits
processor (`ras_logits_processor.py` + `top_p=0.8`; vLLM's own `repetition_penalty` can't be used — it
CUDA-asserts on `prompt_embeds`). The baseline also uses the fluid "pro" reference voice so zh pacing
≈ English. `(full story: P18.)`

### 4.5 Traditional Chinese → Simplified before synth
**Symptom (fixed).** CosyVoice's text frontend garbles long **Traditional** zh (noise past ~10 chars)
while the same sentence in Simplified is flawless; this pipeline feeds Traditional (llama-4-scout /
sherpa output). **Fix:** `COSYVOICE_T2S=1` converts Traditional → Simplified with OpenCC `t2s` before
synthesis (`_to_simplified()`), covering `/tts` + stream + warmup. **Inaudible** — T and S are the same
spoken Mandarin. `(full story: P43.)`

### 4.6 Leading-breath trim
`COSYVOICE_TRIM_LEAD=1` — CosyVoice prepends a low-level breath (0.23 s median, up to 0.60 s) before the
first zh word; the avatar lip-syncs off a Whisper of the waveform so the mouth moves over it. Trimmed
**server-side on whole tensors** (frame RMS, not per-sample abs — a breath spikes above any sample
threshold while staying inaudible): lead-in 0.23 s → a deterministic 0.03 s. A client-side byte-stream
trim was rejected (crashed on odd-sized chunks — see §1) — server-side is the only safe place. `(full
story: P34.)`

### 4.7 Model baseline
`COSYVOICE_MODEL=v2` is what `.env` runs (CosyVoice2-0.5B). `v3` (Fun-CosyVoice3-0.5B, +flow-TRT) is the
selectable accelerated alternative. Switching **requires relaunching the WSL TTS server** (the config
panel's CosyVoice-model card does this) — a plain `.env` edit + pipeline Restart is not enough. `(full
story: P42.)`

---

## 5. LLM & STT

### 5.1 Pin OpenRouter to Groq + `llama-4-scout`
**Symptom (fixed).** The default transpacific route made the LLM hop the dominant TTFO cost plus all
its variance (7–8 s tails). **Fix:** `OPENROUTER_PROVIDER_ONLY=Groq` (injected as
`extra_body.provider.only`) cut the hop to ~0.7 s tight and killed the tail. Model
`meta-llama/llama-4-scout` (Groq, non-reasoning): as fast as llama-3.3-70b, clean substantive
Traditional zh, ~5× cheaper. `OPENROUTER_MAX_TOKENS=500` caps reply length (the fuller zh prompt needs
5–8 sentences without clipping; the old ~50 s monologues were a *different* model ignoring brevity).
Judge model quality with an isolated probe. `(full story: P21.)`

### 5.2 sherpa STT + the invisible pre-t0 second
`STT_PROVIDER=sherpa` (local offline streaming zipformer, bilingual zh-en, ~0 VRAM, zh→Traditional via
OpenCC). **Symptom (fixed):** a flat ~1.0 s dead wait every turn that the TTFO metric couldn't see
(TTFO's stopwatch starts *at* t0). **Root cause:** after Smart Turn returns `COMPLETE`, the strategy
waits `ttfs_p99_latency − stop_secs` for the STT final; sherpa declared no value → pipecat's 1.0 s
cloud default, though sherpa emits its final *synchronously* with the endpoint. **Fix:**
`kwargs.setdefault("ttfs_p99_latency", 0.1)` in `sherpa_stt.py`: `COMPLETE→t0` 1.0 s → 0.09 s, content
unaffected. If you swap STT, declare its **real** measured value. `(full story: P54.)`

---

## 6. Transport & WebRTC

### 6.1 Pin ICE to the Tailscale interface
`WEBRTC_ICE_SUBNET=100.64.0.0/10` — the intermittent remote mic ("works sometimes") was WebRTC ICE
candidate pollution (hyper-v/radmin interfaces). Pinning ICE to the Tailscale 100.64/10 range fixes it.
`(full story: P4.)`

### 6.2 Public link + TURN for strangers
**Default is now `WEBRTC_PUBLIC=0`** (2026-07-18, P57): =1 advertises STUN + Cloudflare TURN, and ICE
then WAITS on those servers to gather relay candidates (~3.5s client cap + ~5s server answer gather) —
pure waste for a tailnet/local/LAN peer, which connects direct `host/host`. Flip to =1 only for a genuine
public non-Tailscale internet link. See §6.6.
`WEBRTC_PUBLIC=1` advertises STUN so an off-tailnet browser reaches the media; the front door is a
Cloudflare quick tunnel (`scripts/tunnel.ps1`, auto-started by `launch.ps1`) carrying only the page +
`/api/offer` signaling, never the media. For a visitor behind symmetric-NAT/UDP-restricted networks a
relay is required: `TURN_CLOUDFLARE` (default ON when public + no static `TURN_URLS`) fetches a fresh
zero-signup relay per connection from Cloudflare's speed-test endpoint (5-min cache, silent STUN-only
fallback). When public, `_restrict_ice_to_subnet` keeps a **set** {Tailscale 100.64/10 + the
internet-facing default-route /32} — pinning to either alone breaks the other's clients. Still
single-client + unauth. `(full story: P38.)`

### 6.3 Phone loudspeaker + jitter/bitrate
- `CLIENT_FORCE_SPEAKER=1` — phone browsers play the voice on the loudspeaker, not the earpiece
  (Android Chrome flips to ear-style routing while the mic is live; iOS gets a WebAudio fallback).
  Mobile-UA only; served to `/studio` via `GET /client/ice-config`. `(full story: P24.)`
- `CLIENT_JITTER_BUFFER_MS=150` (raise to 250–400 for a WAN viewer) and
  `WEBRTC_VIDEO_BITRATE_MAX=500000` (caps aiortc's VP8 ceiling so the video fits a remote link).

### 6.4 Never block the event loop
One asyncio loop carries uvicorn, aiortc RTP send/receive, the pipecat pipeline **and** the MuseTalk
websocket pump. A synchronous `urllib`/`requests` call inside any async handler doesn't just stall that
request — it **stops the loop**: no packets out, no audio pumped, the live call freezes for the whole
timeout. Use `aiohttp` or `run_in_executor`, and cache failures so a dead endpoint isn't re-probed per
request. `(full story: P47.2.)`   

### 6.5 The transport floor is ~0.13 s (not a lever)
Transport + encode + network is **at its physical floor (~0.13 s)** — the measure harness once invented
~0.2 s of phantom "network" that was never on the wire, and its listed lever (`WEBRTC_VIDEO_BITRATE_MAX`)
caps VP8 *video* while the voice is a separate OPUS track, so it could never have moved it. Don't
optimise this row. `(full story: P55.)`

### 6.6 Connect latency — four costs, none the pipeline
"/studio takes ~20s to Connect" was four independent things, found with the always-on **`[connect-timing]`
beacon** (`/studio` POSTs `{cfgMs,overlayMs,micMs,gatherMs,gatherCapped,offerMs,connectMs,pair,rttMs}` to
`POST /client/connect-timing` on 'connected' → one log line, readable from a phone). Local ICE/DTLS is ~30ms,
so the "~5s handshake" folklore was NOT it. The four:
1. **Per-connect pipeline rebuild** — pipecat rebuilds the pipeline every connection, reloading the STT
   models from disk (~1.6–2.7s). Cached process-wide (`sensevoice_stt.py` `_MODEL_CACHE`): build#2 = 0ms.
2. **`localhost` IPv6 trap on :8002** — the avatar server is IPv4-only (`0.0.0.0`); on Windows `localhost`
   tries `::1` first and wastes ~2s per request (hit the `/health` check AND the ws connect). `config.py`
   normalizes `avatar_url`→`127.0.0.1`. urllib/websockets to `localhost:8002` = 2031ms vs 0ms. **#1+#2:
   build→ready ~5.9s → ~0.2s warm.**
3. **STUN/TURN gather wait** — see §6.2; `WEBRTC_PUBLIC=0` removes it (was ~3.5s client + ~5s server).
4. **The 594KB overlay on the connect path** — split-mode's background PNG was `await`ed before the offer
   and re-downloaded every connect; over a ~570 kbps DERP-relayed tailnet it took ~8s AND saturated the link
   so the offer POST stalled ~7s. Now **prefetched once at page load, cached in memory, applied in `ontrack`
   — off the connect path** (`studio_client/index.html` `fetchOverlayData()`). Reconnects are instant.

Result: local instant, Tailscale reconnect ~0.2–0.6s (was ~20s). Note: a ~570 kbps `host/host` link means the
tailnet is likely **DERP-relayed, not direct** — that also makes the avatar *video* choppy (separate from
connect; check `tailscale status` for "relay"). `(full story: P57.)`

---

## 7. Chat / Studio client

The custom client + avatar presets live at **`/studio/`** (`local_services/studio_client/`) — a
self-contained vanilla-JS page speaking the same SmallWebRTC signaling as the prebuilt `/client/`
(which stays the untouched fallback, unsupported under `MUSETALK_SPLIT=1`). Open it **with the trailing
slash**. Extras are two thin endpoints: `POST /client/say {text}` injects a typed turn; `GET
/client/transcript?since=N` feeds the chat bubbles from a read-only observer. **Single-connection:** a
new `/api/offer` disconnects the previous session.

### 7.1 Dedupe transcript frames on `frame.id`, NEVER `id(frame)`
**Symptom (fixed).** The chat bubble silently dropped characters (`自然語言處理` → `自言處理`) while the
voice was perfect. **Root cause:** the observer deduped on `id(frame)` — the *memory address* — and
CPython recycles a freed frame's address into the next frame, so new tokens hit the "already seen" set.
**Fix:** dedupe on pipecat's monotonic unique `frame.id`; reset `_seen` at `LLMFullResponseEndFrame`.
`(full story: P45.)`

### 7.2 Transcript poll must be re-entrant-safe AND idempotent
**Symptom (fixed).** Identical duplicate bubbles. **Root cause:** `setInterval(poll, 200)` fires whether
or not the last poll returned; one >200 ms main-thread hitch sends the next poll with the stale `since`,
the server returns the same rows twice, both render. **Fix:** keep the `polling` in-flight flag AND the
`it.seq > transcriptSeq` skip. `(full story: P46.)`

### 7.3 One bubble per turn, streamed live
The user's speech streams into a live bubble word-by-word (STT interims → `_partial` → 200 ms poll),
then commits as **one** bubble per turn (segments accumulate, commit at `LLMFullResponseStart`). The
user commit keys on the frame **type**, NOT `frame.finalized` (Deepgram's/sherpa's streaming path leaves
it False → gating on it dropped every user bubble). The mic button **mutes** (toggles the track), it
doesn't disconnect. `(full story: P37.)`

### 7.4 Mid-turn interrupt (barge-in)
Under `ALLOW_INTERRUPTIONS=1` (baseline) a mid-reply interruption flushes the current turn clean: on the
`InterruptionFrame` the avatar client drops in-flight server frames (`_flushing`) and the MuseTalk server
drains its `out_q` on `reset`, so the avatar no longer keeps lip-moving silently or leaks the old turn
into the next. A typed turn barges in too — `/client/say` emits an `InterruptionFrame` before the
append. The partial reply is preserved in context (pipecat's aggregator commits it on interruption).
`(full story: P44.)`

### 7.5 No-mic fallback
If `getUserMedia` fails (classic case: an RDP session without audio-recording redirection), the client
connects anyway: warns "No microphone found", sends `audio` as **recvonly**, shows "Type to talk", and
the chat box remains the input. To get a real mic over RDP: mstsc → Local Resources → Remote audio →
Settings → "Record from this computer".

### 7.6 Avatar presets
One GPU runs one avatar, so `/studio/` shows whichever **preset** is live (`AVATAR_PRESET`, config-panel
"Avatar preset" card). A preset = a full backend swap (portrait `AVATAR_REF` + cloned voice
`COSYVOICE_PROMPT_WAV/TEXT` + `LANGUAGE`): `nimbus` (female weather, en) / `leo` (his face + voice, zh —
the live default). A preset relaunches CosyVoice → avatar → pipeline in the P15 load order; a voice
reference must be **Simplified** zh with an accurate transcript. ~2–4 min to switch.

### 7.7 Not-a-bug caveat: mic echo on speakers
On speakers the live mic transcribes the avatar's own voice (`ECHO_GUARD=0` baseline) → gibberish "user"
turns and, under `ALLOW_INTERRUPTIONS=1`, a mid-sentence truncation. **Use headphones.** Duplicate
bubbles with *identical* text = the poll race (§7.2); a *gibberish* second query bubble = mic echo.

---

## 8. Measurement — reading the harness honestly

`python -m scripts.measure --turns 5` drives N turns and writes an ~11-stage **mic-to-ear** waterfall
(Capture → STT-finalize → LLM TTFB → LLM→TTS flush → TTS TTFB → avatar render → steady lead-hold →
transport → browser jitter → browser playout). Median mic-to-ear is ~2.91 s (`docs/LATENCY-MATRIX.md`).

Two things the raw TTFO metric and the harness get wrong — internalise them before trusting a row:

- **TTFO can't see pre-t0.** Its stopwatch starts *at* t0 (user-stopped), so it never counts the cost
  of *deciding* t0 (VAD hangover + Smart Turn end-of-turn). Felt delay = pre-t0 + post-t0. That blind
  spot hid a full second (§5.2). **Judge pre-t0 only from `--observe` on real speech** — the synthetic
  clip's comma pause makes Smart Turn re-poll and invents a ~1.6 s cost real speech doesn't have.
- **Check a row against its physics floor before optimising it.** A row 13× over its floor is the
  *instrument*, not the system — that rule caught five false findings (phantom transport network, a
  fake "session-degradation" bug from `--btail` shorter than the ~50 s reply, a lost first turn from
  `--blead` shorter than the ~5 s ICE handshake). `(full story: P55/P56.)`

`pipeline/metrics.py` (the TtfoMeter) is deliberately untouched — the waterfall is derived in
`scripts/measure/`.
