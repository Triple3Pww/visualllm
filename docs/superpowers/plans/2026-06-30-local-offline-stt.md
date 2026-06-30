# Local Offline STT (SenseVoice on CPU) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fully-offline, CPU/RAM Mandarin (zh-TW) STT — SenseVoice-Small served locally — as an opt-in `STT_PROVIDER=funasr`, keeping Deepgram the default; plus low-risk VRAM-tightening.

**Architecture:** A FastAPI server (`local_services/funasr_server/app.py`, own `funasr-stt` conda env, `:8004`) loads SenseVoice-Small on CPU and returns OpenCC-Traditional text for a posted utterance. A Pipecat `SegmentedSTTService` subclass (`local_services/funasr_stt.py`, mirroring `cosyvoice_tts.py`) posts the buffered utterance and emits a `TranscriptionFrame`. `pipeline/stages/stt.py` gains a `deepgram|funasr` switch.

**Tech Stack:** Pipecat 1.3.0, FunASR + SenseVoice-Small, OpenCC (`s2twp`), FastAPI/uvicorn, aiohttp.

## Global Constraints

- **Verification uses this repo's real tools, NOT a new pytest suite** (repo CLAUDE.md: "no build/lint/unit-test suite — don't invent one"): `python -m scripts.preflight`, a server smoke script, `python -m scripts.measure`, and small standalone `python -c` checks for pure helpers.
- **ASCII-only in all `.py` and `.ps1` source.** Markdown may use full Unicode.
- **Default unchanged:** `STT_PROVIDER` defaults to `deepgram`; offline STT is opt-in. Public `main` behavior must not change unless the env opts in.
- **STT runs on CPU (`FUNASR_DEVICE=cpu`), ~0 VRAM.** Do not place it on the GPU (card is full).
- **OpenCC conversion happens in the server**, so the pipeline env gains no new dependency.
- Servers read their knobs from the **OS env only** (no python-dotenv), like the CosyVoice/MuseTalk servers.
- Degrade gracefully: a down `:8004` or empty transcript yields nothing — never crash a turn.
- Branch: `feat/offline-stt-sensevoice` (already created off `main`). Frequent commits.

---

### Task 1: Config knobs + STT provider switch

**Files:**
- Modify: `pipeline/config.py` (add fields near the existing `# --- STT (Deepgram) ---` block ~line 59)
- Modify: `pipeline/stages/stt.py` (wrap the current Deepgram builder in a provider switch)
- Modify: `.env.example` (document the four knobs)

**Interfaces:**
- Produces: `Config.stt_provider: str`, `Config.funasr_url: str`, `Config.funasr_model: str`, `Config.funasr_device: str`; `build_stt(cfg)` returns a Deepgram service for `deepgram` and a `FunasrSTTService` for `funasr`.

- [ ] **Step 1: Add config fields** — in `pipeline/config.py`, after `deepgram_api_key` (~line 60):

```python
    # --- STT provider switch (deliberate fallback switch, like TTS_PROVIDER) ---
    # deepgram = cloud streaming (default, interim partials); funasr = local OFFLINE
    # SenseVoice-Small on CPU (zh-TW via server-side OpenCC), ~0 VRAM. One flip reverts.
    stt_provider: str = (_get("STT_PROVIDER", "deepgram") or "deepgram").lower()
    funasr_url: str = _get("FUNASR_URL", "http://localhost:8004") or "http://localhost:8004"
    funasr_model: str = _get("FUNASR_MODEL", "iic/SenseVoiceSmall") or "iic/SenseVoiceSmall"
    funasr_device: str = _get("FUNASR_DEVICE", "cpu") or "cpu"
```

- [ ] **Step 2: Wrap the builder in `pipeline/stages/stt.py`** — replace the body of `build_stt` so the existing Deepgram code is the `deepgram` branch and `funasr` builds the local service. Keep the language mapping for Deepgram:

```python
def build_stt(cfg: Config):
    if cfg.stt_provider == "funasr":
        # Local OFFLINE SenseVoice-Small on CPU (~0 VRAM). The server returns
        # Traditional (zh-TW) text via OpenCC, so no pipeline-side conversion.
        from local_services.funasr_stt import FunasrSTTService

        return FunasrSTTService(base_url=cfg.funasr_url)

    from pipecat.services.deepgram.stt import DeepgramSTTService

    if cfg.is_thai:
        language = "th"
    elif cfg.is_mandarin:
        language = "zh-TW"
    else:
        language = "en-US"
    return DeepgramSTTService(
        api_key=cfg.deepgram_api_key,
        settings=DeepgramSTTService.Settings(
            model="nova-2-general",
            language=language,
            smart_format=True,
        ),
    )
```

- [ ] **Step 3: Document the knobs in `.env.example`** — add near the STT section:

```ini
# --- STT provider ---
# deepgram (default, cloud, interim partials) | funasr (local OFFLINE SenseVoice-Small on CPU)
STT_PROVIDER=deepgram
FUNASR_URL=http://localhost:8004        # the local SenseVoice server (start it when STT_PROVIDER=funasr)
FUNASR_MODEL=iic/SenseVoiceSmall
FUNASR_DEVICE=cpu                       # cpu = ~0 VRAM (recommended on the shared GPU)
```

- [ ] **Step 4: Verify config parses and default is unchanged**

Run: `python -c "from pipeline.config import Config; c=Config(); print(c.stt_provider, c.funasr_url, c.funasr_device)"`
Expected: `deepgram http://localhost:8004 cpu`

- [ ] **Step 5: Commit**

```bash
git add pipeline/config.py pipeline/stages/stt.py .env.example
git commit -m "feat(stt): STT_PROVIDER switch + funasr config knobs (default unchanged)"
```

---

### Task 2: SenseVoice server (`local_services/funasr_server/`)

**Files:**
- Create: `local_services/funasr_server/__init__.py` (empty)
- Create: `local_services/funasr_server/app.py`
- Create: `local_services/funasr_server/requirements.txt`
- Create: `local_services/funasr_server/_smoke.py` (standalone smoke check)

**Interfaces:**
- Produces: `POST /stt` (body: raw 16 kHz mono int16 PCM bytes) -> `{"text": "<traditional zh>"}`; `GET /health` -> `{"status": "ok", "model": "<id>", "device": "cpu"}`.
- Consumes (OS env): `FUNASR_MODEL` (default `iic/SenseVoiceSmall`), `FUNASR_DEVICE` (default `cpu`).

- [ ] **Step 1: Write `requirements.txt`**

```
funasr>=1.1.0
torch                # CPU build is fine (FUNASR_DEVICE=cpu)
torchaudio
opencc-python-reimplemented
fastapi
uvicorn[standard]
numpy
```

- [ ] **Step 2: Write `app.py`** — load once at startup; convert PCM->float32; transcribe; OpenCC `s2twp`:

```python
"""Local OFFLINE STT server: SenseVoice-Small via FunASR, on CPU (~0 VRAM).

Mirrors the CosyVoice/MOSS local-server pattern. Serves :8004. The pipeline reaches it
via FUNASR_URL when STT_PROVIDER=funasr. Returns Traditional (zh-TW) text (OpenCC s2twp)
so the pipeline needs no conversion. Reads FUNASR_MODEL / FUNASR_DEVICE from the OS env
ONLY (no python-dotenv), like the other servers.

Run (in the `funasr-stt` conda env):
    python -m uvicorn local_services.funasr_server.app:app --host 0.0.0.0 --port 8004
If model download hits the conda cert store, set SSL_CERT_FILE to certifi's cacert.pem.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Request

MODEL_ID = os.environ.get("FUNASR_MODEL", "iic/SenseVoiceSmall")
DEVICE = os.environ.get("FUNASR_DEVICE", "cpu")

_state: dict = {}


def _pcm16_to_float32(pcm: bytes) -> np.ndarray:
    """16 kHz mono int16 PCM bytes -> float32 [-1, 1] mono, the form FunASR expects."""
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


@asynccontextmanager
async def lifespan(app: FastAPI):
    from funasr import AutoModel
    from opencc import OpenCC

    # Warm the model at startup so the first turn isn't penalized.
    _state["model"] = AutoModel(model=MODEL_ID, device=DEVICE, disable_update=True)
    _state["s2tw"] = OpenCC("s2twp")  # Simplified -> Traditional (Taiwan, with phrases)
    print(f"[funasr] SenseVoice ready: {MODEL_ID} on {DEVICE}", flush=True)
    yield
    _state.clear()


app = FastAPI(title="Local SenseVoice STT", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok" if "model" in _state else "loading", "model": MODEL_ID, "device": DEVICE}


@app.post("/stt")
async def stt(request: Request):
    pcm = await request.body()
    if not pcm:
        return {"text": ""}
    audio = _pcm16_to_float32(pcm)
    # SenseVoice: language="auto" detects zh; use_itn adds punctuation/inverse-text-norm.
    res = _state["model"].generate(input=audio, cache={}, language="auto", use_itn=True)
    raw = res[0]["text"] if res else ""
    # SenseVoice prefixes rich tags like <|zh|><|NEUTRAL|>...; strip them, then s2twp.
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
    clean = rich_transcription_postprocess(raw)
    return {"text": _state["s2tw"].convert(clean)}
```

- [ ] **Step 3: Write `_smoke.py`** — exercises the PCM helper + (if the model is loaded) a silent buffer:

```python
"""Standalone smoke check for the SenseVoice server helpers. Run with the system python
(tests the pure helper) or post a wav to a running :8004 server.

    python -m local_services.funasr_server._smoke           # unit: PCM conversion
"""
import numpy as np

from local_services.funasr_server.app import _pcm16_to_float32


def test_pcm_conversion():
    pcm = np.array([0, 16384, -16384, 32767], dtype=np.int16).tobytes()
    out = _pcm16_to_float32(pcm)
    assert out.dtype == np.float32
    assert out.shape == (4,)
    assert abs(out[1] - 0.5) < 1e-3 and abs(out[2] + 0.5) < 1e-3
    print("PCM conversion OK:", out)


if __name__ == "__main__":
    test_pcm_conversion()
    print("smoke OK")
```

- [ ] **Step 4: Verify the pure helper passes (no model/env needed)**

Run: `python -m local_services.funasr_server._smoke`
Expected: `PCM conversion OK: [...]` then `smoke OK`

- [ ] **Step 5: Verify the server boots + transcribes (manual, in the `funasr-stt` env)** — one-time, requires the env + model download:

```bash
# In the funasr-stt conda env (creating it + the model download are documented in Task 4):
python -m uvicorn local_services.funasr_server.app:app --host 0.0.0.0 --port 8004 &
curl -s http://localhost:8004/health           # -> {"status":"ok",...}
# Post a real 16k mono PCM utterance and confirm Traditional text comes back:
python -c "import requests,wave; w=wave.open('output/q_ai.wav','rb'); \
  print(requests.post('http://localhost:8004/stt', data=w.readframes(w.getnframes())).json())"
```
Expected: `/health` ok; `/stt` returns `{"text": "<traditional chinese>"}`.

- [ ] **Step 6: Commit**

```bash
git add local_services/funasr_server/
git commit -m "feat(stt): SenseVoice-Small offline STT server (:8004, CPU, OpenCC s2twp)"
```

---

### Task 3: Pipecat STT wrapper (`local_services/funasr_stt.py`)

**Files:**
- Create: `local_services/funasr_stt.py`
- Create: `local_services/_funasr_stt_check.py` (standalone mocked check)

**Interfaces:**
- Consumes: `POST /stt` from Task 2 (raw PCM -> `{"text": ...}`); `build_stt` from Task 1 calls `FunasrSTTService(base_url=...)`.
- Produces: `class FunasrSTTService(SegmentedSTTService)` with `__init__(self, *, base_url: str, **kwargs)`; overrides `async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]` yielding a `TranscriptionFrame`.

- [ ] **Step 1: Write `funasr_stt.py`** — mirror `cosyvoice_tts.py`'s aiohttp/session style:

```python
"""Pipecat STT wrapper for the local SenseVoice server (Task 2). A SegmentedSTTService:
Pipecat buffers the utterance between VAD start/stop, then calls run_stt(audio) once.
We POST the raw PCM to FUNASR_URL/stt and emit the Traditional text it returns. Mirrors
local_services/cosyvoice_tts.py (the local-server client pattern). Degrades gracefully:
a down server or empty transcript yields nothing -- never crashes the turn."""
from __future__ import annotations

from typing import AsyncGenerator

import aiohttp
from loguru import logger
from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601


class FunasrSTTService(SegmentedSTTService):
    def __init__(self, *, base_url: str, **kwargs):
        super().__init__(**kwargs)
        self._base_url = base_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def stop(self, frame):
        if self._session and not self._session.closed:
            await self._session.close()
        await super().stop(frame)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        yield None  # SegmentedSTTService contract: first yield is a None heartbeat
        try:
            session = await self._get_session()
            async with session.post(f"{self._base_url}/stt", data=audio) as resp:
                resp.raise_for_status()
                text = (await resp.json()).get("text", "").strip()
        except Exception as e:  # server down / network -> degrade, don't crash the turn
            logger.warning(f"FunASR STT call failed: {e}")
            return
        if text:
            yield TranscriptionFrame(text, "", time_now_iso8601())
```

- [ ] **Step 2: Write `_funasr_stt_check.py`** — mock the HTTP call, assert a `TranscriptionFrame` with the server text is yielded, and that a failure yields nothing:

```python
"""Standalone check for FunasrSTTService with a mocked server. Run:
    python -m local_services._funasr_stt_check
No network/model needed."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from pipecat.frames.frames import TranscriptionFrame

from local_services.funasr_stt import FunasrSTTService


def _fake_post(text):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = AsyncMock(return_value={"text": text})
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.post = MagicMock(return_value=cm)
    return session


async def _collect(svc):
    out = []
    async for f in svc.run_stt(b"\x00\x00" * 100):
        if f is not None:
            out.append(f)
    return out


def test_emits_transcription():
    svc = FunasrSTTService(base_url="http://x")
    with patch.object(svc, "_get_session", AsyncMock(return_value=_fake_post("天氣晴朗"))):
        frames = asyncio.run(_collect(svc))
    assert len(frames) == 1 and isinstance(frames[0], TranscriptionFrame)
    assert frames[0].text == "天氣晴朗"
    print("emits TranscriptionFrame OK")


def test_failure_yields_nothing():
    svc = FunasrSTTService(base_url="http://x")
    failing = AsyncMock(side_effect=RuntimeError("down"))
    with patch.object(svc, "_get_session", failing):
        frames = asyncio.run(_collect(svc))
    assert frames == []
    print("graceful-degrade OK")


if __name__ == "__main__":
    test_emits_transcription()
    test_failure_yields_nothing()
    print("all checks OK")
```

- [ ] **Step 3: Verify the wrapper logic (no network/model)**

Run: `python -m local_services._funasr_stt_check`
Expected: `emits TranscriptionFrame OK`, `graceful-degrade OK`, `all checks OK`

- [ ] **Step 4: Verify the full stack imports via preflight with the funasr provider**

Run: `STT_PROVIDER=funasr python -m scripts.preflight` (PowerShell: `$env:STT_PROVIDER='funasr'; python -m scripts.preflight`)
Expected: imports resolve; the STT stage builds the funasr service without ImportError.

- [ ] **Step 5: Commit**

```bash
git add local_services/funasr_stt.py local_services/_funasr_stt_check.py
git commit -m "feat(stt): Pipecat SegmentedSTTService wrapper for the SenseVoice server"
```

---

### Task 4: Run wiring + docs (server startup, env setup, INSTALL/WORKFLOW)

**Files:**
- Modify: `scripts/run.ps1` (start `:8004` when `STT_PROVIDER=funasr`)
- Modify: `INSTALL.md` (offline-STT section), `WORKFLOW.md` (knob reference), `CLAUDE.md` (stack note)

- [ ] **Step 1: Read `scripts/run.ps1`** to match its style for launching a server + reading `.env`.

Run: `cat scripts/run.ps1`

- [ ] **Step 2: Add an optional `:8004` start in `run.ps1`** — gated on `STT_PROVIDER=funasr` read from `.env`, launched in the `funasr-stt` conda env (ASCII-only source; mirror how it starts the avatar server). Start it BEFORE the pipeline so `/health` is up. Example shape (adapt to the file's actual helpers):

```powershell
# Optional local OFFLINE STT (SenseVoice on CPU) -- only when STT_PROVIDER=funasr.
if ($sttProvider -eq 'funasr') {
    Start-Process -WindowStyle Minimized -FilePath $funasrPython `
      -ArgumentList '-m','uvicorn','local_services.funasr_server.app:app','--host','0.0.0.0','--port','8004'
    # (waits on http://localhost:8004/health the same way the CosyVoice wait works)
}
```

- [ ] **Step 3: Add the offline-STT setup to `INSTALL.md`** — a subsection under STT: create the
  `funasr-stt` conda env, `pip install -r local_services/funasr_server/requirements.txt`, the model
  auto-downloads on first run (note the `SSL_CERT_FILE`=certifi gotcha), set `STT_PROVIDER=funasr` +
  `FUNASR_URL`, and the honest tradeoff (offline, ~0 VRAM, segmented, +~0.3-1.5s on CPU, zh-TW via OpenCC).

- [ ] **Step 4: Add the knobs to `WORKFLOW.md` and a one-line note to `CLAUDE.md`** — `STT_PROVIDER`,
  `FUNASR_URL`, `FUNASR_MODEL`, `FUNASR_DEVICE` in the `.env` reference; CLAUDE.md stack table/notes gain
  the `funasr` STT option (default still Deepgram).

- [ ] **Step 5: Verify docs reference the knobs**

Run: `grep -l "STT_PROVIDER" INSTALL.md WORKFLOW.md CLAUDE.md .env.example`
Expected: all four files listed.

- [ ] **Step 6: Commit**

```bash
git add scripts/run.ps1 INSTALL.md WORKFLOW.md CLAUDE.md
git commit -m "docs+run(stt): launch :8004 when STT_PROVIDER=funasr; document the offline-STT path"
```

---

### Task 5: VRAM-tightening (investigate + easy wins)

**Files:**
- Modify: `scripts/run.ps1` (set `PYTORCH_CUDA_ALLOC_CONF`)
- Modify: `CLAUDE.md` / `WORKFLOW.md` (document the KV-trim knobs + findings)
- Create: `docs/gpu-memory-notes.md` (the investigation record)

- [ ] **Step 1: Investigate what holds VRAM** — with the stack stopped vs running:

```bash
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader
nvidia-smi   # full table: note which PIDs (vLLM, MuseTalk, browser, stale runs) hold memory
```
Record the per-state numbers (idle / cosyvoice-only / + musetalk / + a turn) in `docs/gpu-memory-notes.md`,
and flag any reclaimable stale/other-app memory (the 254-MiB-free reading may be partly non-pipeline).

- [ ] **Step 2: Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** in `run.ps1` for the avatar +
  pipeline processes (ASCII-only). This reduces allocator fragmentation so the same workload fits in less
  reserved VRAM. Comment the why.

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'   # cut CUDA allocator fragmentation
```

- [ ] **Step 3: Document the vLLM KV-trim knobs** in `CLAUDE.md`/`WORKFLOW.md` and `docs/gpu-memory-notes.md`:
  `COSYVOICE_VLLM_GPU_UTIL` (default 0.3; safe floor ~0.25 -- 0.2 crashed) and `--max-model-len` (TTS
  sequences are short; lowering it frees KV-cache VRAM). **Document only -- do not change the tuned default.**

- [ ] **Step 4: Verify the stack still starts with the alloc flag + measure free VRAM**

Run: `.\scripts\run.ps1` then `nvidia-smi --query-gpu=memory.free --format=csv,noheader`
Expected: stack boots; record whether `expandable_segments` increased free VRAM. (No regression to TTFO.)

- [ ] **Step 5: Commit**

```bash
git add scripts/run.ps1 CLAUDE.md WORKFLOW.md docs/gpu-memory-notes.md
git commit -m "perf(gpu): expandable_segments + document KV-trim knobs + VRAM investigation notes"
```

---

### Task 6: End-to-end live verification

- [ ] **Step 1: Bring up the offline stack** — `.env`: `STT_PROVIDER=funasr`, `LANGUAGE=zh`; start CosyVoice, then `run.ps1` (which now also starts `:8004`).

- [ ] **Step 2: Confirm a zh-TW turn end-to-end** — open `http://localhost:7860/client/`, speak Mandarin, confirm the transcript (Traditional characters) drives the LLM->TTS->avatar, and capture the `[TTFO]` line.

- [ ] **Step 3: Measure the latency cost** — `python -m scripts.measure --offline-capture`; record the STT contribution vs the Deepgram baseline in `docs/gpu-memory-notes.md` (or a short note in the spec).

- [ ] **Step 4: Confirm VRAM unchanged by STT** — `nvidia-smi` with `STT_PROVIDER=funasr` running shows no new GPU process for SenseVoice (it's on CPU). Record it.

- [ ] **Step 5: Commit any doc updates** from the measured numbers.

```bash
git add -A && git commit -m "docs(stt): record measured offline-STT latency + zero-VRAM confirmation"
```

---

## Self-Review

- **Spec coverage:** server (T2), wrapper (T3), provider switch+config (T1), run/launcher (T4), docs incl. INSTALL/WORKFLOW/CLAUDE/.env (T1,T4), OpenCC-in-server (T2), VRAM wins (T5), live verification (T6). All spec sections covered.
- **Placeholder scan:** all code steps show full code; verification steps give exact commands + expected output. The only "adapt to the file" note (T4.2 run.ps1) is gated by reading the file first in T4.1 — unavoidable since run.ps1's helpers aren't quoted here, but the shape + intent are concrete.
- **Type consistency:** `FunasrSTTService(base_url=...)` is identical in T1 (caller), T3 (def), and the check; `/stt` contract (raw PCM -> `{"text"}`) matches between T2 (server) and T3 (client); `run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]` matches the verified Pipecat base signature.
- **Repo convention honored:** no new pytest suite; verification via preflight + standalone `python -m ...` checks + measure harness.
