# Per-Frame Audio Coupling (proto 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every MuseTalk server→client frame self-describing — `kind` (real/held/idle) + `audio_pos` (cumulative 16k samples consumed to render it) — so the steady-mode client releases voice paired to what the server *actually rendered*, replacing the `i/fps` index arithmetic and the P39 byte-identical held-frame heuristic.

**Architecture:** A 16-byte binary header (`MTF2` magic + kind + audio_pos) is prepended to every ws frame, but ONLY after the client requests `proto: 2` in its `config` message and the server acks with a `{"type":"proto","v":2}` marker. Old harnesses (`_capture.py`, `_drive_frames.py`, `_capture_synced.py`, archive tests) never request it and keep the bare-frame wire unchanged. The delivered voice stays the client-buffered ORIGINAL sample-rate TTS audio (24k) — the header carries a *position*, not audio bytes, so there is no 16k quality downgrade (the OpenAvatarChat bytes-in-packet design would downgrade us; this is the metadata translation of the same idea).

**Tech Stack:** Python (system 3.11 pipeline / musetalk conda server), FastAPI websockets, struct-packed header.

## Global Constraints

- The steady-mode invariants MUST survive: synced start via server lead-prime (`MUSETALK_LEAD_FRAMES`), burst feed (`MUSETALK_FEED_BURST_S`), interrupt flush (P44 `_flushing` + `reset` drain), close crossfade (P12), odd-byte carries (`_align_even`, `_srv_carry`, P3/P40), split mode (frame-content agnostic).
- `live` mode (audio-master) and the `_unsynced` fallback are untouched.
- Proto 1 (bare frames) must remain byte-identical for clients that don't request proto 2.
- ASCII-safe `.py` source (cp1252 console).
- Comments state the *why* (house style).

## Header format (shared by Tasks 1–2)

```python
# 16 bytes, little-endian: magic 4s | kind u8 | 3 pad | audio_pos u64
# kind: 0 = REAL lip-synced frame (audio_pos valid: cumulative 16k samples of THIS
#           turn's audio covered once this frame is shown, pad excluded)
#       1 = HELD re-send (render underflow or lead-prime; audio_pos = last real pos)
#       2 = IDLE/neutral (between turns / END_TAIL; audio_pos = 0)
FRAME_HDR = struct.Struct("<4sB3xQ")
FRAME_MAGIC = b"MTF2"
```

---

### Task 1: Server — self-describing frames behind `proto: 2`

**Files:**
- Modify: `local_services/musetalk_server/app.py` (stream handler ~750–1003)
- Test: `C:\Users\MARU\AppData\Local\Temp\...\scratchpad\proto2_probe.py` (new, throwaway)

**Interfaces:**
- Consumes: existing `engine.render_segment(seg) -> list[bytes]`, `engine.samples_for_frames`.
- Produces: ws binary = `FRAME_HDR + rgb` when proto 2 negotiated; text marker `{"type":"proto","v":2}` sent once, right after a `config` containing `"proto": 2`. `out_q` items become `(bytes, kind:int, pos:int)` tuples internally.

- [ ] **Step 1: Write the failing probe** — `proto2_probe.py`: connect to `ws://localhost:8002/stream`, send `{"type":"config","fps":12,"proto":2}`, expect the `proto` ack marker within 2s, then feed a wav as 16k PCM after `speech_start`, collect frames for ~8s, assert: every binary message starts with `MTF2`; kinds ∈ {0,1,2}; real-frame `audio_pos` strictly increasing; final real pos == total samples fed (± one frame of ceil pad).

```python
"""proto2 probe: asserts the self-describing frame contract against :8002."""
import asyncio, json, struct, sys, wave
import numpy as np, websockets

HDR = struct.Struct("<4sB3xQ")
WAV = r"E:\Claude\VisualLLm\output\_zh_q_def.wav"   # 16k mono already

async def main():
    wf = wave.open(WAV, "rb"); sr = wf.getframerate()
    pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    if sr != 16000:
        idx = np.linspace(0, len(pcm) - 1, int(len(pcm) * 16000 / sr))
        pcm = np.interp(idx, np.arange(len(pcm)), pcm).astype(np.int16)
    ws = await websockets.connect("ws://localhost:8002/stream", max_size=None)
    await ws.send(json.dumps({"type": "config", "fps": 12, "proto": 2}))
    ack = json.loads(await asyncio.wait_for(ws.recv(), 2))          # FAILS on old server
    assert ack.get("type") == "proto" and ack.get("v") == 2, ack
    await ws.send(json.dumps({"type": "speech_start"}))
    for i in range(0, len(pcm), 3200):
        await ws.send(pcm[i:i+3200].tobytes()); await asyncio.sleep(0.1)
    await ws.send(json.dumps({"type": "speech_end"}))
    kinds, last_pos, n_real = set(), -1, 0
    end = asyncio.get_event_loop().time() + 8
    while asyncio.get_event_loop().time() < end:
        try: m = await asyncio.wait_for(ws.recv(), 1)
        except asyncio.TimeoutError: continue
        if isinstance(m, bytes):
            magic, kind, pos = HDR.unpack(m[:16])
            assert magic == b"MTF2", magic
            kinds.add(kind)
            if kind == 0:
                assert pos > last_pos, (pos, last_pos); last_pos = pos; n_real += 1
    assert 0 in kinds, "no real frames seen"
    fed = len(pcm)
    assert abs(last_pos - fed) <= 16000 // 12 + 1, (last_pos, fed)
    print(f"OK: {n_real} real frames, kinds={sorted(kinds)}, final pos {last_pos}/{fed}")
    await ws.close()

asyncio.run(main())
```

- [ ] **Step 2: Run it against the unchanged server** — Expected: FAIL (`TimeoutError` or first `recv` is a bare frame, magic assert fires).

- [ ] **Step 3: Implement.** In `app.py`:
  - Top of file (near other constants): add `FRAME_HDR`/`FRAME_MAGIC` (code in the Header section above) + `import struct`.
  - In `stream()`: add `proto = 1` and `turn_pos = 0` locals; in the `config` branch, `if int(evt.get("proto", 1)) >= 2: proto = 2; await _mark({"type": "proto", "v": 2})`.
  - `enqueue(frame, kind=0, pos=0)` — store tuples `(frame, kind, pos)`; `render(segment, real_len)` distributes positions linearly across the segment's frames ending at `turn_pos + real_len` (pad excluded), advances `turn_pos`, enqueues each frame with its pos:

```python
    async def render(segment: np.ndarray, real_len: int | None = None) -> int:
        nonlocal turn_pos
        real = len(segment) if real_len is None else real_len
        async with _render_lock:
            frames = await asyncio.to_thread(engine.render_segment, segment)
        n = len(frames)
        for k, f in enumerate(frames):
            # Cumulative REAL samples covered once frame k is shown. Linear within the
            # segment, exact at its end -- pos is the SOURCE OF TRUTH the client pairs
            # audio against, so it must never include the speech_end zero-pad.
            pos = turn_pos + (real * (k + 1)) // n if n else turn_pos
            enqueue(f, 0, pos)
        turn_pos += real
        return n
```

  - `speech_start` branch: `turn_pos = 0`.
  - `speech_end` final partial segment: call `render(seg, real_len=len(audio_buf))` (pre-pad length). END_TAIL neutrals: `enqueue(engine.neutral_frame(), 2, 0)`.
  - In `pump()`: track `last_kind`/`last_pos` beside `last`; a real dequeue sends kind 0 + its pos; the prime/hold resend path sends kind 1 + `last_pos`; idle/neutral sends kind 2 + 0. ONE send helper so proto 1 stays bare:

```python
        async def send_frame(fb: bytes, kind: int, pos: int) -> None:
            if proto >= 2:
                await ws.send_bytes(FRAME_HDR.pack(FRAME_MAGIC, kind, pos) + fb)
            else:
                await ws.send_bytes(fb)
```

  (Pump reads `proto` from the enclosing scope each call, so a config arriving after the pump starts is honored — same pattern as the per-tick fps re-read.)

- [ ] **Step 4: Restart the avatar server, run the probe** — Expected: PASS (`OK: ...`).
- [ ] **Step 5: Run `python -m scripts._drive_frames output/reply_concise.wav 12`** (proto 1 client) — Expected: unchanged behavior, REAL rendered = audio_sec*fps ±1.
- [ ] **Step 6: Commit** — `feat(avatar-server): self-describing frames (kind + audio_pos) behind proto 2`.

### Task 2: Client — position-paired release

**Files:**
- Modify: `local_services/musetalk_video.py`

**Interfaces:**
- Consumes: proto-2 wire from Task 1 (`FRAME_HDR`, ack marker).
- Produces: unchanged downstream contract (TTSAudioRawFrame passthrough + tagged OutputImageRawFrame).

- [ ] **Step 1: Implement.**
  - Module top: `import struct`; `FRAME_HDR = struct.Struct("<4sB3xQ")`; `FRAME_MAGIC = b"MTF2"`.
  - `__init__`: `self._proto2 = False`; `self._vpos: list[int] = []`.
  - `_open_ws`: send `{"type": "config", "fps": self._fps, "proto": 2}`.
  - `_on_marker`: `elif kind == "proto": self._proto2 = int(evt.get("v", 1)) >= 2` (+ reset to `False` in `_open_ws` before sending, so a reconnect to an old server downgrades cleanly).
  - `_on_frame(self, img)`: at the top, parse and strip the header when `self._proto2`:

```python
        kind = None                       # None = proto-1 bare frame (heuristics stay)
        if self._proto2 and len(img) >= 16 and img[:4] == FRAME_MAGIC:
            _m, kind, pos = FRAME_HDR.unpack(img[:16])
            img = img[16:]
```

  - Held detection: `is_dup` becomes explicit when the header is present, heuristic otherwise:

```python
        is_dup = (self._sync and self._video_active and not self._unsynced
                  and ((kind == 1) if kind is not None
                       else (bool(self._vbuf) and img == self._vbuf[-1])))
```

  - Buffering (under the lock): `else: self._vbuf.append(img); self._vpos.append(pos if kind == 0 else 0)`.
  - `_advance`: the audio-cap guard only applies to proto-1 index pairing (a proto-2 real frame's pos can never exceed audio the client already buffered — `_abuf.append` happens in the same `process_frame` call that queues the feed):

```python
            target = len(self._vbuf)
            if not (self._proto2 and self._vpos):
                audio_cap = math.ceil(self._audio_clock_s * self._fps) + 1
                target = min(target, audio_cap)
```

  - `_emit_pair`: pair by the server-declared position when present, index arithmetic otherwise:

```python
        ft = (self._vpos[i] / MUSETALK_SR
              if self._proto2 and i < len(self._vpos) and self._vpos[i] > 0
              else i / self._fps)
```

  - `_reset_turn`: `self._vpos = []`.
- [ ] **Step 2: `python -m scripts.preflight`** — Expected: all imports resolve.
- [ ] **Step 3: Archive regressions** — run `python archive\_sync_routing_test.py`, `python archive\_screech_repro_test.py`, `python archive\_interrupt_flush_test.py`, `python archive\_frame_deficit_repro_test.py` — Expected: all PASS (they exercise proto-1/transport paths that must not change).
- [ ] **Step 4: Commit** — `feat(avatar-client): pair voice release to server-declared audio_pos (proto 2)`.

### Task 3: Live verification (the eye's proxy first, then the eye)

- [ ] **Step 1: Restart the stack** (avatar server + pipeline via `scripts/run.ps1`; TTS stays up).
- [ ] **Step 2: Confirm negotiation** — pipeline log shows the `proto` ack path taken (add nothing; check `[stream] config` on the server + no `MTF2` errors), and `[musetalk sync] hold=` lines look normal.
- [ ] **Step 3: `python -m scripts.measure --mic output/_zh_q_def.wav` ×2** — Expected: TTFO within baseline (2.53–3.84s), `Received video` ~12 fps, `Freeze` < 500ms, `[avatar timing]` end drift ≤ ±0.15s, `held/dup` counter still sane.
- [ ] **Step 4: Interrupt path** — drive one measure run, then a second connection immediately (the probe reconnect exercises single-client supersede); check no stale frames leak (`_flushing` path logs clean).
- [ ] **Step 5: The user's eye** — the final arbiter (P19 both directions): a real browser session on /studio/ to confirm lipsync feels unchanged-or-better. (Flag for the user; cannot be automated.)

### Task 4: Docs + commit

- [ ] **Step 1:** CLAUDE.md wire-contract bullet: add the proto-2 line (self-describing frames, kind+audio_pos, negotiated, proto-1 fallback for the offline harnesses).
- [ ] **Step 2:** `docs/PROBLEMS-AND-FIXES.md`: short entry — not a bug fix, but P39/P47.3's structural closure; note the OpenAvatarChat provenance + why bytes-in-packet was rejected (16k voice downgrade).
- [ ] **Step 3:** Commit docs; update the project memory note.

## Self-Review

- Spec coverage: header format ✓, negotiation ✓, server pos accounting incl. speech_end pad ✓, held/prime/idle kinds ✓, client parse + pairing + cap removal ✓, proto-1 compatibility (old harnesses) ✓, regression gates ✓, live verify ✓.
- Types consistent: `enqueue(bytes,int,int)`, `render(seg, real_len)`, `_vpos: list[int]`, `FRAME_HDR` identical both sides.
- No placeholders: every step has code or an exact command.
