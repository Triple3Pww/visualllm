# Mouth-crop overlay: pristine static background + streamed mouth patch

**Date:** 2026-07-11
**Status:** design, pending implementation
**Goal:** Make the avatar *picture* look sharp by delivering the background portrait once as a
pristine, never-video-compressed still and streaming only the small moving mouth region over the
existing WebRTC video track. The `/nimbus` client composites the two.

## Motivation

Today the MuseTalk server composites the rendered mouth into the full portrait and streams the
**whole frame** over WebRTC VP8. VP8 is temporal (it already sends mostly the moving region), so the
win here is **not** raw bandwidth — it is **sharpness**: the static background stops being
VP8-compressed, and the encoder's whole budget lands on the tiny mouth patch.

### Hard limit (expectation-setting, not a bug)
MuseTalk generates the animated face at a **fixed 256px** (why `MUSETALK_SIZE=512` "only sharpens the
STATIC frame, not the animated mouth" — see `CLAUDE.md`). So:
- **Background / portrait at rest → big, real sharpness win** (pristine hi-res still).
- **Animated mouth patch → small win only.** Removing VP8 blur helps, but the source is 256px; it
  will never be photo-crisp. That needs a different avatar model, out of scope here.

## Scope

- **In:** `MUSETALK_SPLIT=1` mode; server sends the bbox mouth crop instead of the full frame; a
  one-time overlay-assets handshake (background PNG + bbox); `/nimbus` canvas compositing.
- **Out:** the prebuilt `/client` (it only plays a video track and cannot composite — it is
  unsupported while `MUSETALK_SPLIT=1`, and stays the untouched full-frame fallback at `=0`).
- **Out:** any change to the A/V-sync machinery, TTS, LLM, or STT. The sync path is reused verbatim.

## Key insight that keeps this cheap and low-risk

The entire hard-won A/V-sync machinery in `musetalk_video.py` (steady mode, `video_clock` pacing,
held-dup drop, close crossfade) is **agnostic to frame content** — it pins rendered frame *N* to audio
at *N/fps*. Feeding it a small mouth crop instead of a full frame changes **nothing** in that code.
So the risky part of the codebase is not touched; we only change (a) what pixels the server puts in
each frame, (b) the frame dimensions, and (c) how `/nimbus` draws them.

Because the server's `_composite()` already blends the mouth crop's edges toward the original portrait
(`get_image_blending`), the blended crop pasted opaquely over that same portrait is seamless — **no
per-frame coordinates or client-side mask needed** for a static portrait (fixed bbox). If a seam shows
against the *hi-res* background at the resample boundary, we add a one-time alpha feather mask (also
sent in the handshake). Start without it.

## Architecture

Three thin changes, all behind `MUSETALK_SPLIT` (default `0`, i.e. today's behavior):

### 1. MuseTalk server (`local_services/musetalk_server/app.py`)
- **Crop-mode frame output.** When split mode is on, every frame the pump sends (`ws.send_bytes`) is
  the **bbox crop** `combine[y1:y2, x1:x2]`, **resized to a fixed square `MUSETALK_SPLIT_SIZE`
  (default 256)** instead of the full 512 frame. The fixed square keeps the VP8 track dimensions
  stable and known to the pipeline from env (no startup fetch to learn a per-portrait crop size); the
  client un-stretches the square back into the bbox rect on draw, and since MuseTalk's face is 256px
  the resize loses ~nothing. Idle/neutral frames become the same fixed-size mouth crop of the neutral
  portrait. `_composite` gains a "return the fixed-size crop, not the full frame" path; the pump loop
  is otherwise unchanged (markers, held-last, neutral rest all still work — the bytes are just smaller).
- **`GET /overlay-assets`** (new). Returns the one-time compositing assets the client needs:
  `{ bg_png (base64, the pristine hi-res portrait = frame_cycle[0] at MUSETALK_BASE_MAX res),
     bbox: [x1,y1,x2,y2] in bg-image pixels, bg_size: [W,H], crop_size: [w,h] }`. Derived from the
  already-computed `frame_cycle[0]` + `coord_cycle[0]` — so it is automatic for any uploaded portrait.

### 2. Pipeline (`pipeline/main.py`, `pipeline/stages/…`)
- Set the WebRTC `video_out_width/height` to **`MUSETALK_SPLIT_SIZE`** (a square, from env — no
  startup fetch) instead of `avatar_size` when split mode is on. `MuseTalkVideoService` already
  forwards whatever bytes/size it is told — pass it `(split_size, split_size)`. One-fps-everywhere and
  `video_out_is_live = not sync_av` invariants are unchanged.
- **`GET /client/avatar-overlay`** (new, same `_inject_client_patches` pattern as `/client/transcript`):
  proxies the server's `/overlay-assets` to `/nimbus`. `no-store`.

### 3. `/nimbus` client (`local_services/nimbus_client/index.html`)
- On connect, `fetch('/client/avatar-overlay')`. Draw the `bg_png` into a full-stage `<canvas>` (or an
  `<img>` behind a canvas). Keep the WebRTC video track in an **offscreen** `<video>`.
- A `requestAnimationFrame` loop draws the background once, then `drawImage(video, …)` into the bbox
  rect (scaled from `crop_size` → the bbox's on-screen size) every frame. Optional alpha feather via a
  mask sent in the handshake, only if a seam appears.
- Fallback: if `/client/avatar-overlay` 404s or `MUSETALK_SPLIT=0`, keep today's behavior (video track
  fills the stage). So `/nimbus` degrades gracefully.

## Data flow (one turn, split mode)

```
[once, on connect]  /nimbus --GET /client/avatar-overlay--> pipeline --GET /overlay-assets--> :8002
                    /nimbus caches bg_png + bbox, paints background canvas

[per turn]  TTS audio --> MuseTalkVideoService --ws--> :8002 renders mouth (256px) ->
            blend -> CROP to bbox -> ws.send_bytes(crop) --> MuseTalkVideoService (sync UNCHANGED)
            -> OutputImageRawFrame(size=crop) -> WebRTC VP8 track (budget on the mouth)
            -> /nimbus offscreen <video> -> rAF loop draws crop over pristine bg canvas, in sync
```

## Validation (the "is it a good idea?" gate)

Per the user's hard-won lesson (*capture the REAL delivered output; the probe passes what the eye
rejects*), validation is the **live `/nimbus` view**, not an offline sim: run the stack with
`MUSETALK_SPLIT=1`, open `/nimbus`, and judge by eye whether the crisp background + 256px mouth patch
looks better than today's all-512 full frame. Success = the picture reads sharper and the mouth patch
has **no visible seam** and stays in A/V sync (the sync path is reused, so sync should be identical).

## Risks / open questions

- **Seam at the hi-res boundary.** Blended-crop-over-same-portrait is seamless at equal res; against a
  *hi-res* background the resample boundary might show. Mitigation: alpha feather mask (one-time),
  already accounted for in the handshake shape.
- **VP8 dimensions.** Solved by the fixed square `MUSETALK_SPLIT_SIZE=256` (even, stable) — the crop
  is resized to it server-side and un-stretched into the bbox rect client-side, so no per-portrait
  padding math is needed.
- **Idle-motion mode.** With `MUSETALK_IDLE_MOTION=1` the portrait sways, so the bbox moves per frame —
  the fixed-bbox assumption breaks. Phase 1 targets the default `MUSETALK_IDLE_MOTION=0` (static
  portrait). Split mode + idle motion is out of scope (document it; the server can refuse or fall back).

## Long-term (already mostly free)
"Any portrait I upload auto-does this" needs no extra work: the server derives bbox + portrait from
`AVATAR_REF` during its existing one-time preparation, so `/overlay-assets` is correct for any image.
The only future item is supporting `MUSETALK_IDLE_MOTION=1` (moving bbox → per-frame coords).
