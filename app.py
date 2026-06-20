"""
Phase 5 — Production TTS API (FastAPI) around the local CosyVoice2 engine,
plus a simple demo web interface.

  GET  /           -> demo web page (templates/index.html)
  POST /tts        {"text": "...", "speed": 1.0}  -> audio/wav
  GET  /health     -> {"status": "ok", ...}
  GET  /info       -> JSON service metadata
  GET  /static/*   -> css / js assets

Run:
  /opt/anaconda3/envs/tts/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8001

Bound to port 8001 to match the avatar pipeline's COSYVOICE_URL. Designed for
the LLM -> TTS -> MuseTalk chain: a teammate's LLM service POSTs text here and
streams the returned wav into MuseTalk. The web page is for demo/testing only
and calls the same POST /tts endpoint.
"""
import io
import os
import uuid
import time

import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from tts_engine import get_engine

HERE = os.path.dirname(__file__)
OUTDIR = os.path.join(HERE, "outputs")
STATIC_DIR = os.path.join(HERE, "static")
TEMPLATES_DIR = os.path.join(HERE, "templates")
os.makedirs(OUTDIR, exist_ok=True)

_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    _engine = get_engine()  # load model once at startup
    yield


app = FastAPI(title="Local CosyVoice2 TTS", version="1.0", lifespan=lifespan)

# Serve CSS/JS assets for the demo page.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize (zh-TW or English)")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Playback speed multiplier")


@app.get("/")
def home():
    """Demo web interface."""
    return FileResponse(os.path.join(TEMPLATES_DIR, "index.html"))


@app.get("/info")
def info():
    return {"service": "Local CosyVoice2 TTS", "endpoints": ["/ (web)", "/tts (POST)", "/health"]}


@app.get("/health")
def health():
    if _engine is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok", "device": _engine.device, "sample_rate": _engine.sample_rate}


@app.post("/tts")
def tts(req: TTSRequest):
    if _engine is None:
        raise HTTPException(status_code=503, detail="engine still loading")
    try:
        t0 = time.perf_counter()
        wav, sr = _engine.synthesize(req.text, speed=req.speed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001 — surface engine failures to the caller
        raise HTTPException(status_code=500, detail=f"synthesis failed: {e}")

    path = os.path.join(OUTDIR, f"tts_{uuid.uuid4().hex}.wav")
    torchaudio.save(path, wav, sr)
    gen = time.perf_counter() - t0
    dur = wav.shape[1] / sr
    return FileResponse(
        path,
        media_type="audio/wav",
        filename="tts.wav",
        headers={
            "X-Generation-Seconds": f"{gen:.3f}",
            "X-Audio-Seconds": f"{dur:.3f}",
            "X-RTF": f"{(gen / dur) if dur else 0:.3f}",
        },
    )
