# Mouth-crop Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the avatar picture sharp by sending the background portrait once as a pristine still and streaming only the fixed-size mouth crop over the existing WebRTC track, composited in `/nimbus`.

**Architecture:** Behind `MUSETALK_SPLIT=1`, the MuseTalk server (`:8002`) emits a fixed `MUSETALK_SPLIT_SIZE` (256²) mouth crop per frame instead of the full 512 frame, and exposes `GET /overlay-assets` (pristine background PNG + bbox). The pipeline sizes the WebRTC video track to 256² and proxies the assets at `/client/avatar-overlay`. `/nimbus` fetches the assets once, paints the crisp background, and canvas-composites each mouth-crop video frame into the bbox rect. The A/V-sync machinery is reused verbatim (it is frame-content-agnostic).

**Tech Stack:** Python 3.11 (pipeline, system env), `musetalk` conda env (avatar server), FastAPI + websockets, aiortc/VP8 (pipecat SmallWebRTC), vanilla JS + `<canvas>` (`/nimbus`), OpenCV (`cv2`).

## Global Constraints

- **No unit-test suite; do not invent one** (`CLAUDE.md`). Verify by running the real servers, `python -m scripts.preflight`, driving a turn, and eyeballing `/nimbus`.
- **`.py` server source must stay ASCII-safe** (Windows console is cp1252); use `--`/`->` in comments, not `—`/`→`.
- **Default OFF / fully revertible:** `MUSETALK_SPLIT` default `0` reproduces today's exact behavior; `/client` prebuilt stays the untouched full-frame fallback.
- **One fps everywhere is load-bearing** and **`video_out_is_live = not avatar_sync_with_audio`** — do not change either invariant.
- **Do not touch** `musetalk_video.py`'s sync logic, TTS/LLM/STT stages, `metrics.py`.
- **Phase 1 targets `MUSETALK_IDLE_MOTION=0`** (static portrait, fixed bbox). Split mode force-disables the idle loop.
- Frame bytes on the wire are raw **RGB** (`_frame_to_bytes` does BGR->RGB). `frame_cycle[i]` and `coord_cycle[i]` are **BGR** and native-portrait pixel space (up to `MUSETALK_BASE_MAX=768`).

---

### Task 1: MuseTalk server — crop-mode output + `/overlay-assets`

**Files:**
- Modify: `local_services/musetalk_server/app.py` (engine `__init__` / `prepare` area near `self._neutral` at :180; `_composite` at :496-507; `neutral_frame` at :417-418; `_build_idle_loop` at :439-448; new route near `@app.get("/health")` at :905)

**Interfaces:**
- Consumes: `engine.frame_cycle[0]` (BGR portrait), `engine.coord_cycle[0]` (`(x1,y1,x2,y2)` bbox), `engine.mask_cycle`/`mask_coords_cycle` (blend), `get_image_blending`.
- Produces (for Task 2/3): `GET /overlay-assets` -> JSON `{"bg_png": "<base64>", "bbox": [x1,y1,x2,y2], "bg_size": [W,H], "split_size": N}`. When `MUSETALK_SPLIT=1`, every `ws.send_bytes` frame is `N*N*3` RGB bytes (N = `split_size`).

- [ ] **Step 1: Add split config + cached crop assets to the engine**

Near the module env reads at top (`BASE_MAX = ...` at :79), add:

```python
SPLIT = os.getenv("MUSETALK_SPLIT", "0").lower() in ("1", "true", "yes")
SPLIT_SIZE = int(os.getenv("MUSETALK_SPLIT_SIZE", "256"))
```

In the engine, right after `self._neutral = self._frame_to_bytes(self.frame_cycle[0])` (:180), precompute the neutral crop and cache the split bbox (guard so a portrait with no face is unaffected):

```python
        # --- split mode (MUSETALK_SPLIT): stream only the fixed-size mouth crop ---
        self._split = SPLIT
        self._split_size = SPLIT_SIZE
        self._split_bbox = self.coord_cycle[0] if self.coord_cycle else None
        self._neutral_split = (
            self._crop_split(self.frame_cycle[0]) if self._split and self._split_bbox else None
        )
        if self._split:
            logger.info(f"[split] MUSETALK_SPLIT on: streaming {self._split_size}px mouth crop "
                        f"(bbox={self._split_bbox}).")
```

- [ ] **Step 2: Add the `_crop_split` helper**

Add a method on the engine (place it just above `_composite` at :496):

```python
    def _crop_split(self, frame_bgr) -> bytes:
        """Fixed-size RGB mouth crop for split mode: crop the bbox region of a full BGR
        frame and resize to split_size square. The client un-stretches it into the bbox
        rect. Keeps the VP8 track a stable known size (the pipeline sets it from env)."""
        import cv2
        x1, y1, x2, y2 = self._split_bbox
        crop = frame_bgr[y1:y2, x1:x2]
        out = cv2.resize(crop, (self._split_size, self._split_size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
```

- [ ] **Step 3: Make `_composite` and `neutral_frame` return the crop in split mode**

In `_composite` (:496), replace the final `return self._frame_to_bytes(combine)` (:507) with:

```python
        if self._split and self._split_bbox is not None:
            return self._crop_split(combine)   # fixed-size mouth crop (split mode)
        return self._frame_to_bytes(combine)
```

In `neutral_frame` (:417), return the neutral crop when split:

```python
    def neutral_frame(self) -> bytes:
        if self._split and self._neutral_split is not None:
            return self._neutral_split
        return self._neutral
```

- [ ] **Step 4: Force-disable the idle loop in split mode**

At the top of `_build_idle_loop` (:439, right after `import cv2`), add:

```python
        if getattr(self, "_split", False):
            # Split mode targets a STATIC portrait (fixed bbox); an idle sway would move the
            # bbox and break the fixed-crop mapping. Rest on the neutral mouth crop instead.
            self._idle_loop = []
            return
```

- [ ] **Step 5: Add the `/overlay-assets` route**

Just above `@app.get("/health")` (:905), add:

```python
@app.get("/overlay-assets")
def overlay_assets():
    """One-time compositing assets for the split-mode /nimbus client: the pristine
    background portrait + the bbox the mouth crop maps into. Derived from the already-
    prepared frame_cycle[0]/coord_cycle[0], so it is automatic for any AVATAR_REF image."""
    import base64
    import cv2
    from fastapi.responses import JSONResponse
    if not getattr(engine, "_split", False) or engine._split_bbox is None:
        return JSONResponse({"split": False}, status_code=404)
    bg = engine.frame_cycle[0]                       # BGR native portrait
    h, w = bg.shape[:2]
    ok, buf = cv2.imencode(".png", bg)               # cv2 writes correct color from BGR
    if not ok:
        return JSONResponse({"error": "encode"}, status_code=500)
    x1, y1, x2, y2 = [int(v) for v in engine._split_bbox]
    return JSONResponse({
        "split": True,
        "bg_png": base64.b64encode(buf.tobytes()).decode("ascii"),
        "bbox": [x1, y1, x2, y2],
        "bg_size": [int(w), int(h)],
        "split_size": int(engine._split_size),
    })
```

- [ ] **Step 6: Verify the server alone (crop size + assets)**

Start the avatar server in split mode (its own conda env), then check both surfaces:

```bash
# terminal A -- server (musetalk env). MUSETALK_IDLE_MOTION=0 keeps the bbox static.
MUSETALK_SPLIT=1 MUSETALK_SPLIT_SIZE=256 MUSETALK_IDLE_MOTION=0 \
  E:/miniconda3/envs/musetalk/python.exe -u -m local_services.musetalk_server.app
# terminal B -- once it logs "MuseTalk ready" and "[split] MUSETALK_SPLIT on":
curl -s http://localhost:8002/overlay-assets | python -c "import sys,json;d=json.load(sys.stdin);print({k:(v if k!='bg_png' else f'<{len(v)}b64>') for k,v in d.items()})"
# Expected: {'split': True, 'bg_png': '<...b64>', 'bbox': [..4 ints..], 'bg_size': [W,H], 'split_size': 256}
```

Drive one turn through the ws and confirm each delivered frame is `256*256*3 = 196608` bytes (a fixed-size crop, not `512*512*3`):

```bash
E:/miniconda3/envs/musetalk/python.exe -m local_services.musetalk_server._capture output/q_ai.wav
# The offline capture renders frames from the ws; it must complete without error.
# If _capture hardcodes a frame size, instead add a one-off print of len(message) in a scratch
# ws client; the assertion is: every binary frame == split_size**2 * 3 bytes.
```

Expected: `/overlay-assets` returns the JSON above; every streamed binary frame is `196608` bytes.

- [ ] **Step 7: Commit**

```bash
git add local_services/musetalk_server/app.py
git commit -m "feat(avatar): MUSETALK_SPLIT server -- stream fixed-size mouth crop + /overlay-assets"
```

---

### Task 2: Pipeline — config knobs, video-track sizing, overlay proxy

**Files:**
- Modify: `pipeline/config.py` (near `avatar_size` at :163-168)
- Modify: `pipeline/stages/avatar.py` (:24-31)
- Modify: `pipeline/main.py` (video_out sizing at :228-232; new GET branch in `_inject_client_patches` near :570)

**Interfaces:**
- Consumes: `GET http://localhost:8002/overlay-assets` (Task 1).
- Produces (for Task 3): `config.avatar_split: bool`, `config.avatar_split_size: int`; WebRTC video track sized `split_size` square when split; `GET /client/avatar-overlay` -> the server's `/overlay-assets` JSON (or 404 when off).

- [ ] **Step 1: Add config properties**

In `pipeline/config.py`, right after the `avatar_size` property (ends :168), add:

```python
    @property
    def avatar_split(self) -> bool:
        """Split mode (MUSETALK_SPLIT): the avatar server streams only the mouth crop and
        /nimbus composites it over a pristine still. Default off (full-frame, /client works)."""
        return (_get("MUSETALK_SPLIT", "0") or "0").lower() in ("1", "true", "yes", "on")

    @property
    def avatar_split_size(self) -> int:
        """Fixed square px of the streamed mouth crop in split mode (MUSETALK_SPLIT_SIZE).
        MUST equal the avatar server's value; the WebRTC track is sized to it."""
        return int(_get("MUSETALK_SPLIT_SIZE", "256") or "256")
```

- [ ] **Step 2: Size the video service to the crop in split mode**

In `pipeline/stages/avatar.py`, change the `size`/return (:27-31) to:

```python
    size = cfg.avatar_size
    _warn_if_server_down(cfg.avatar_url)

    if cfg.avatar_split:
        s = cfg.avatar_split_size
        logger.info(f"Avatar: MuseTalk SPLIT at {cfg.avatar_url} (fps={fps}, crop={s}px)")
        return MuseTalkVideoService(base_url=cfg.avatar_url, fps=int(fps), image_size=(s, s))

    logger.info(f"Avatar: local MuseTalk at {cfg.avatar_url} (output fps={fps}, size={size})")
    return MuseTalkVideoService(base_url=cfg.avatar_url, fps=int(fps), image_size=(size, size))
```

- [ ] **Step 3: Size the WebRTC video track to the crop in split mode**

In `pipeline/main.py`, the `video_out_width`/`video_out_height` lines (:231-232) currently use `config.avatar_size`. Replace those two lines with a size chosen from split mode. Add just above the `transport_params = {` dict (after :218 `sync_av = ...`):

```python
    # Split mode streams a fixed-size square mouth crop (config.avatar_split_size); the WebRTC
    # track MUST match those frame dimensions. Off = the full square portrait (avatar_size).
    _vout = config.avatar_split_size if config.avatar_split else config.avatar_size
```

Then change lines :231-232 to:

```python
            video_out_width=_vout,
            video_out_height=_vout,
```

- [ ] **Step 4: Add the `/client/avatar-overlay` proxy**

In `pipeline/main.py`, inside `_inject_client_patches`, add a new GET branch next to the `/client/transcript` branch (after :581). It proxies the server assets so `/nimbus` (same-origin) can fetch them:

```python
        # Nimbus split-mode overlay: proxy the avatar server's one-time compositing assets
        # (pristine background PNG + bbox) so /nimbus can paint the crisp background and
        # composite the streamed mouth crop over it. 404 when MUSETALK_SPLIT is off.
        if request.method == "GET" and request.url.path == "/client/avatar-overlay":
            import aiohttp
            url = config.avatar_url.rstrip("/") + "/overlay-assets"
            try:
                async with aiohttp.ClientSession() as sess:
                    async with sess.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                        body = await r.read()
                        return HTMLResponse(body.decode("utf-8", "replace"),
                                            status_code=r.status,
                                            media_type="application/json")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[avatar-overlay] proxy failed: {e!r}")
                return HTMLResponse('{"split": false}', status_code=502,
                                    media_type="application/json")
```

- [ ] **Step 5: Verify imports + proxy resolve**

```bash
python -m scripts.preflight
# Expected: exits 0, no import errors (confirms main.py/config.py/avatar.py still import clean).
```

With the Task-1 server still running in split mode, start the pipeline in split mode and hit the proxy:

```bash
# terminal C -- pipeline (SYSTEM python). It reaches :8002 for assets.
MUSETALK_SPLIT=1 MUSETALK_SPLIT_SIZE=256 MUSETALK_IDLE_MOTION=0 python -m pipeline.main
# terminal D:
curl -s http://localhost:7860/client/avatar-overlay | python -c "import sys,json;d=json.load(sys.stdin);print(d.get('split'), d.get('bbox'), d.get('bg_size'), d.get('split_size'))"
# Expected: True [..bbox..] [W,H] 256
```

Expected: `preflight` passes; `/client/avatar-overlay` returns the same JSON the server's `/overlay-assets` returns.

- [ ] **Step 6: Commit**

```bash
git add pipeline/config.py pipeline/stages/avatar.py pipeline/main.py
git commit -m "feat(avatar): pipeline split-mode -- size WebRTC track to crop + /client/avatar-overlay proxy"
```

---

### Task 3: `/nimbus` — canvas composite of background + mouth crop

**Files:**
- Modify: `local_services/nimbus_client/index.html` (video element :547; `pc.ontrack` :807-811; teardown :862; add a compositor block)

**Interfaces:**
- Consumes: `GET /client/avatar-overlay` (Task 2) -> `{split, bg_png, bbox:[x1,y1,x2,y2], bg_size:[W,H], split_size}`; the WebRTC video track (now a `split_size` square mouth crop).
- Produces: on-screen composite (pristine background + mouth crop drawn into the bbox). Graceful fallback to today's full-video behavior when `split` is false / assets 404.

- [ ] **Step 1: Add a `<canvas>` compositor over the stage**

Next to the `<video id="avatar" ...>` element (:547), add a sibling canvas (the video becomes the offscreen source in split mode). Keep the video element for the non-split fallback:

```html
        <video id="avatar" class="presenter-img" autoplay playsinline muted></video>
        <canvas id="avatarCanvas" class="presenter-img" style="display:none"></canvas>
```

- [ ] **Step 2: Add the compositor logic**

In the client `<script>`, after `const avatar = document.getElementById('avatar');` (:620), add:

```javascript
    const avatarCanvas = document.getElementById('avatarCanvas');
    let _split = null;      // {bbox, bgW, bgH} once loaded, else null (full-frame fallback)
    let _bgImg = null;      // pristine background <img>
    let _rafId = 0;

    async function loadOverlay() {
      try {
        const r = await fetch('/client/avatar-overlay', { cache: 'no-store' });
        if (!r.ok) return false;
        const d = await r.json();
        if (!d || !d.split) return false;
        _bgImg = new Image();
        await new Promise((res, rej) => {
          _bgImg.onload = res; _bgImg.onerror = rej;
          _bgImg.src = 'data:image/png;base64,' + d.bg_png;
        });
        _split = { bbox: d.bbox, bgW: d.bg_size[0], bgH: d.bg_size[1] };
        return true;
      } catch (e) { console.warn('overlay load failed', e); return false; }
    }

    function startCompositor() {
      // Draw the pristine background, then the mouth-crop video stretched into the bbox rect,
      // every animation frame. Canvas is sized to the background's native pixels; CSS scales
      // the whole stage (background + overlaid crop scale together, so the bbox stays aligned).
      const c = avatarCanvas, ctx = c.getContext('2d');
      c.width = _split.bgW; c.height = _split.bgH;
      const [x1, y1, x2, y2] = _split.bbox;
      avatar.style.display = 'none';
      c.style.display = '';
      const draw = () => {
        if (_bgImg) ctx.drawImage(_bgImg, 0, 0, c.width, c.height);
        if (avatar.readyState >= 2 && avatar.videoWidth > 0) {
          ctx.drawImage(avatar, x1, y1, x2 - x1, y2 - y1);  // un-stretch square crop into bbox
        }
        _rafId = requestAnimationFrame(draw);
      };
      cancelAnimationFrame(_rafId);
      draw();
    }
```

- [ ] **Step 3: Route the video track through the compositor**

In `pc.ontrack` (:807-811), when a split overlay is loaded, keep feeding the same `<video>` (now offscreen) but start the compositor. Change the video branch:

```javascript
        pc.ontrack = (e) => {
          const stream = e.streams[0] || new MediaStream([e.track]);
          if (e.track.kind === 'video') {
            avatar.srcObject = stream; avatar.play().catch(()=>{});
            if (_split) startCompositor();   // draw crop over pristine bg (else <video> shows full frame)
          } else if (e.track.kind === 'audio') {
            botAudio.srcObject = stream; botAudio.play().catch(()=>{});
          }
        };
```

- [ ] **Step 4: Load the overlay before connecting**

Find where the connect flow begins (the `connectBtn`/connect handler that builds `pc` and calls `pc.addTransceiver('video', ...)` near :833). Immediately before creating the peer connection / sending the offer, `await loadOverlay();` so `_split` is known when `ontrack` fires. If it returns false, `_split` stays null and the plain `<video>` full-frame path is used (fallback).

```javascript
      await loadOverlay();   // sets _split if MUSETALK_SPLIT is on; else full-frame <video> fallback
      // ... existing pc = new RTCPeerConnection(...) / addTransceiver / offer code follows ...
```

- [ ] **Step 5: Reset on teardown**

In `teardown` (:862, where `avatar.srcObject = null`), stop the compositor and reset so a reconnect re-evaluates split mode:

```javascript
      avatar.srcObject = null; botAudio.srcObject = null;
      cancelAnimationFrame(_rafId); _rafId = 0;
      avatarCanvas.style.display = 'none'; avatar.style.display = '';
      _split = null; _bgImg = null;
```

- [ ] **Step 6: Verify live in the browser**

With all three processes running in split mode (CosyVoice in WSL, avatar server + pipeline from the steps above), open `http://localhost:7860/nimbus/` (WITH trailing slash), connect, and speak (or type a turn). Confirm by eye:
- The background/portrait looks **crisp** (pristine PNG), noticeably sharper than the all-512 full frame.
- The mouth animates inside the bbox with **no visible rectangular seam** against the background.
- A/V sync is unchanged (lips track the voice like today — the sync path was untouched).

If a seam shows: note it for the follow-up feather-mask step (out of scope for first pass). If the whole stage is blank: check `/client/avatar-overlay` returned `split:true` and the canvas replaced the video (`avatarCanvas` visible, `avatar` hidden).

- [ ] **Step 7: Commit**

```bash
git add local_services/nimbus_client/index.html
git commit -m "feat(nimbus): composite pristine background + streamed mouth crop (split mode)"
```

---

### Task 4: End-to-end verification + docs

**Files:**
- Modify: `.env.example` (or the `.env` reference in `WORKFLOW.md` §8) — document `MUSETALK_SPLIT` / `MUSETALK_SPLIT_SIZE`
- Modify: `CLAUDE.md` (avatar knobs paragraph) and `STATUS.md` (current-state note)

**Interfaces:**
- Consumes: the running split-mode stack from Tasks 1-3.
- Produces: documented knobs; a recorded verdict of the live look-test.

- [ ] **Step 1: Confirm the fallback is intact (regression guard)**

Restart the stack with `MUSETALK_SPLIT=0` (default) and open BOTH `/client/` and `/nimbus/`. Confirm the avatar shows the full-frame portrait and talks exactly as before (this proves the change is fully revertible and `/client` is untouched).

```bash
python -m scripts.preflight   # still 0
# open http://localhost:7860/client/ and /nimbus/ -> full-frame avatar, unchanged.
```

Expected: identical to pre-change behavior on both clients.

- [ ] **Step 2: Record the split-mode measurement (optional, sync sanity)**

Split mode must not change A/V sync (the sync path is reused). Sanity-check with the harness:

```bash
python -m scripts.measure --offline-capture
# TTFO + handoffs should match the pre-change baseline within noise (no new drift).
```

Expected: TTFO/sync figures in line with the current baseline (the change is transport-payload only).

- [ ] **Step 3: Document the knobs**

Add to `CLAUDE.md`'s avatar-knobs section (near `MUSETALK_SIZE`):

```markdown
`MUSETALK_SPLIT` (**0 = default, full-frame**; `1` = stream only the fixed-size mouth crop and let
`/nimbus` composite it over a pristine, never-compressed background still -> crisp picture, encoder
budget concentrated on the mouth. `/nimbus` ONLY -- the prebuilt `/client` can't composite and is
unsupported while on. The animated mouth stays MuseTalk's 256px (a model limit, not transport);
only the background gets truly sharp. Targets `MUSETALK_IDLE_MOTION=0` / a static portrait. Server:
`/overlay-assets`; pipeline proxy: `/client/avatar-overlay`. `docs/superpowers/specs/2026-07-11-mouth-crop-overlay-design.md`),
`MUSETALK_SPLIT_SIZE` (**256** -- the square px of the streamed crop; MUST match server + pipeline).
```

Add a one-line current-state note to `STATUS.md` pointing at the spec + this plan.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md STATUS.md WORKFLOW.md
git commit -m "docs(avatar): document MUSETALK_SPLIT mouth-crop overlay mode"
```

---

## Self-Review

**Spec coverage:**
- Crop-mode frame output (fixed square) -> Task 1 Steps 1-3. ✓
- Idle-motion out of scope / force-disabled -> Task 1 Step 4. ✓
- `/overlay-assets` (bg + bbox, auto for any AVATAR_REF) -> Task 1 Step 5. ✓
- WebRTC track sized to crop; sync untouched -> Task 2 Steps 2-3. ✓
- `/client/avatar-overlay` proxy (same middleware pattern) -> Task 2 Step 4. ✓
- `/nimbus` canvas composite + graceful fallback -> Task 3. ✓
- `/client` unsupported-while-on / full-frame fallback at 0 -> Task 4 Step 1. ✓
- Validation = live `/nimbus` eye + no seam + sync -> Task 3 Step 6, Task 4 Step 2. ✓
- Seam risk / feather-mask deferred -> Task 3 Step 6 note. ✓
- Docs of the knobs -> Task 4 Step 3. ✓

**Placeholder scan:** No TBD/TODO; every code step shows real code; verification uses this repo's real commands (`preflight`, `_capture`, `measure`, `curl`, browser). ✓

**Type consistency:** `_split`/`_split_size`/`_split_bbox`/`_neutral_split` and `_crop_split()` consistent across Task 1. `avatar_split`/`avatar_split_size` consistent across config -> avatar.py -> main.py. `/overlay-assets` JSON keys (`bg_png`, `bbox`, `bg_size`, `split_size`) consistent server -> proxy -> `/nimbus` (`d.bg_png`, `d.bbox`, `d.bg_size`, matching `split_size`). ✓
