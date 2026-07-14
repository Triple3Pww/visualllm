"""JaiTTS-F5TTS Thai STREAMING TTS server -- same wire contract as tts/cosyvoice-server
and the MOSS server (POST /tts/stream {text, voice, sample_rate} -> raw 16-bit PCM mono
stream), so the pipeline reaches it through the existing CosyVoice client just by
repointing the URL. Set TTS_PROVIDER=jaitts (see pipeline/stages/tts.py).

WHY THIS EXISTS: CosyVoice -- the default local TTS -- **cannot speak Thai**. JaiTTS
(hf://JTS-AI/JaiTTS-F5TTS, F5-family, Apache-2.0) is the Thai voice path. This server is
what makes LANGUAGE=th actually speak.

THE CHUNKING RULE (load-bearing -- do not "optimize" it away):
  F5/JaiTTS **degrades on long single generations** -- the flow-matching solver drifts and
  the vocoder loses formant structure on the final syllables, producing a metallic "alien"
  warble. SHORT generations are always clean. So this server splits text into short chunks
  (on Thai sentence-final particles) and synthesizes each separately.
  Verified 2026-07-12; full write-up + the rejected knobs (speed<1.0, nfe 96+, cfg 2.0,
  reference trimming -- all tested WORSE) in visualllm-business/TTS-EMOTION.md §3.
  Bonus: chunking is also what makes this *stream* -- each chunk's PCM goes out as soon as
  it's generated, so TTFB is one short chunk, not the whole utterance.

THE VOICE: a fixed reference clip (JAITTS_REF + JAITTS_REF_TEXT) -- F5 is a zero-shot
cloner, so the reference IS the voice. The request's `voice` field is ignored (same as
CosyVoice/MOSS). Reference rules (8-15s, one speaker, clean, ends on a sentence-final
particle) are in TTS-EMOTION.md §5. Ship an OWNED clip -- the R&D refs are cloned from
third-party video and are not license-clear.

RUN (in the shared F5 venv -- a plain venv, NOT conda):
  cd /e/Claude/visualllm
  JAITTS_REF=E:/Claude/visualllm-business/jaitts/ref_ytm2.wav \
  JAITTS_REF_TEXT=E:/Claude/visualllm-business/jaitts/ref_ytm2.txt \
  E:/f5-spike/.venv-f5/Scripts/python.exe -m uvicorn \
      local_services.jaitts_server.app:app --host 0.0.0.0 --port 8004

Then point the pipeline at it:  TTS_PROVIDER=jaitts  LANGUAGE=th
"""
from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterator

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch  # torch BEFORE flowtts/f5_tts (native-segfault guard)

# torchaudio 2.11 routes .load through torchcodec (needs an ffmpeg shared build it can't
# find) -> bypass with soundfile. Same fix the jaitts spike + F5-THAI server needed.
import soundfile as _sf
import torchaudio


def _load_via_soundfile(filepath, *a, **k):
    data, sr = _sf.read(str(filepath), dtype="float32", always_2d=True)  # [samples, ch]
    return torch.from_numpy(data.T).contiguous(), sr                     # [ch, samples]


torchaudio.load = _load_via_soundfile

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

MODEL_SR = 24000

# The JaiTTS spike lives in the business repo; it vendors thonburian-tts (the `flowtts`
# module JaiTTS requires -- JaiTTS's own repo is eval-only, no synth code).
JAITTS_DIR = os.environ.get("JAITTS_DIR", r"E:/Claude/visualllm-business/jaitts")
JAITTS_REF = os.environ.get("JAITTS_REF", os.path.join(JAITTS_DIR, "ref_ytm2.wav"))
JAITTS_REF_TEXT = os.environ.get("JAITTS_REF_TEXT", os.path.join(JAITTS_DIR, "ref_ytm2.txt"))
MAX_CHARS = int(os.environ.get("JAITTS_MAX_CHARS", "60") or "60")

sys.path.insert(0, os.path.join(JAITTS_DIR, "thonburian-tts"))  # expose `flowtts`

_E: dict = {}
_LOCK = threading.Lock()  # serialize turns (single-client avatar; F5 pipe isn't reentrant)

# Thai sentence-final particles -- the best places to break a line.
BREAKS = ("ค่ะ", "คะ", "ครับ", "นะคะ", "นะฮะ", "นะ", "จ้ะ", "ค่า")


def _hard_split_chunk(s: str, max_chars: int) -> list[str]:
    """Character-window fallback for a chunk that has no spaces to split on.

    Thai has no inter-word spaces, so a chunk that came in as (or accumulated
    into) one long space-less run needs its own cut logic. Prefer cutting
    right after the LATEST BREAKS particle found inside the window (a real
    sentence-final boundary); fall back to a hard cut at the window edge if
    none is present.
    """
    out = []
    while len(s) > max_chars:
        window = s[:max_chars]
        cut = 0
        for p in BREAKS:
            idx = window.rfind(p)
            if idx != -1:
                cut = max(cut, idx + len(p))
        if not cut:
            cut = max_chars
        out.append(s[:cut])
        s = s[cut:]
    if s:
        out.append(s)
    return out


def chunk_thai(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split Thai text into SHORT chunks (the anti-'alien' rule). Mirrors
    visualllm-business/jaitts/synth_clean.py -- keep the two in sync."""
    chunks, cur = [], ""
    for w in text.split():
        cand = (cur + " " + w).strip()
        if len(cand) > max_chars and cur:
            chunks.append(cur)
            cur = w
        else:
            cur = cand
            if any(cur.endswith(p) for p in BREAKS) and len(cur) >= max_chars * 0.5:
                chunks.append(cur)
                cur = ""
    if cur:
        chunks.append(cur)
    if not chunks:
        return [text]
    # Fallback for chunks .split() couldn't break up -- Thai has no inter-word spaces, so a
    # space-less run collapses into ONE giant "word" above and sails straight through the
    # length check untouched. That's the documented long-generation "alien warble" trigger
    # (see the module docstring). Text WITH spaces never produces a chunk this long (the loop
    # above splits every time a chunk crosses max_chars), so this is a no-op there --
    # byte-identical output for spaced text.
    out = []
    for c in chunks:
        if len(c) > max_chars * 2:
            out.extend(_hard_split_chunk(c, max_chars))
        else:
            out.append(c)
    return out


def _load() -> None:
    from flowtts.inference import AudioConfig, FlowTTSPipeline, ModelConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[jaitts] loading JaiTTS-F5TTS on {device} ...", flush=True)
    mc = ModelConfig(
        language="th",
        model_type="F5",
        checkpoint="hf://JTS-AI/JaiTTS-F5TTS/model.pt",
        vocab_file="hf://JTS-AI/JaiTTS-F5TTS/vocab.txt",
        vocoder="vocos",
        device=device,
    )
    # nfe 32 / cfg 2.5 / speed 1.0 -- the tested-good trio. Slower speed drops words and
    # WORSENS the artifact; nfe 96+ costs RTF 0.85-1.06 (breaks the live bar). See §3.
    ac = AudioConfig(cfg_strength=2.5, nfe_step=32, speed=1.0)
    pipe = FlowTTSPipeline(
        model_config=mc, audio_config=ac, temp_dir=os.path.join(JAITTS_DIR, "temp_jaitts")
    )
    ref_text = Path(JAITTS_REF_TEXT).read_text(encoding="utf-8").strip()
    _E.update(pipe=pipe, ref_wav=JAITTS_REF, ref_text=ref_text, tmp=os.path.join(JAITTS_DIR, "temp_jaitts"))
    print(f"[jaitts] ready (ref={JAITTS_REF})", flush=True)


def _trim_edges(d: np.ndarray, sr: int = MODEL_SR) -> np.ndarray:
    """Strip leading/trailing near-silence so chunk joins are tight (dead air reads as lag)."""
    win = int(sr * 0.01)
    if len(d) < win * 2:
        return d
    rms = np.array([np.sqrt(np.mean(d[i:i + win] ** 2) + 1e-12) for i in range(0, len(d) - win, win)])
    on = np.where(rms > rms.max() * 0.04)[0]
    if not len(on):
        return d
    return d[max(0, (on[0] - 2)) * win: min(len(d), (on[-1] + 3) * win)]


def _synth_stream(text: str) -> Iterator[np.ndarray]:
    """Yield float32 mono 24 kHz audio, ONE SHORT CHUNK AT A TIME (streams + stays clean)."""
    pipe, ref_wav, ref_text, tmp = _E["pipe"], _E["ref_wav"], _E["ref_text"], _E["tmp"]
    chunks = chunk_thai(text)
    for i, c in enumerate(chunks):
        out = os.path.join(tmp, f"_srv{i}.wav")
        pipe(text=c, ref_voice=ref_wav, ref_text=ref_text, output_file=out, speed=1.0)
        d, _ = _sf.read(out, dtype="float32")
        if d.ndim > 1:
            d = d.mean(1)
        d = _trim_edges(d)
        f = int(MODEL_SR * 0.015)  # 15ms edge fades -> inaudible seams
        if len(d) > 2 * f:
            d[:f] *= np.linspace(0, 1, f)
            d[-f:] *= np.linspace(1, 0, f)
        yield d
        if i < len(chunks) - 1:  # 120ms breath between chunks (0.3s+ reads as lag)
            yield np.zeros(int(MODEL_SR * 0.12), dtype="float32")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load()
    t0 = time.perf_counter()
    try:  # warm the CUDA path so turn 1 isn't slow
        for _ in _synth_stream("สวัสดีค่ะ"):
            pass
        print(f"[jaitts] warmup done in {time.perf_counter() - t0:.1f}s", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[jaitts] warmup skipped: {e}", flush=True)
    yield


app = FastAPI(title="JaiTTS-F5TTS Thai streaming TTS", version="1.0", lifespan=lifespan)


class TTSStreamRequest(BaseModel):
    text: str
    voice: str = Field("th", description="informational; the engine uses JAITTS_REF")
    sample_rate: int = 24000


@app.get("/health")
def health():
    return {"ok": "pipe" in _E}


@app.get("/")
def root():
    return {
        "service": "JaiTTS-F5TTS (Thai, chunked-streaming)",
        "endpoints": ["/tts/stream (POST)", "/health"],
        "ref": JAITTS_REF,
    }


def _to_pcm16(chunk: np.ndarray, target_sr: int) -> bytes:
    chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
    if target_sr != MODEL_SR:
        import librosa

        chunk = librosa.resample(chunk, orig_sr=MODEL_SR, target_sr=target_sr)
    return (np.clip(chunk, -1.0, 1.0) * 32767).astype("<i2").tobytes()


@app.post("/tts/stream")
def tts_stream(req: TTSStreamRequest):
    if "pipe" not in _E:
        return JSONResponse({"error": "engine not ready"}, status_code=503)

    def gen():
        with _LOCK:  # one turn at a time (single-client avatar)
            for chunk in _synth_stream(req.text):
                pcm = _to_pcm16(chunk, req.sample_rate)
                if pcm:
                    yield pcm

    return StreamingResponse(gen(), media_type="audio/L16")
