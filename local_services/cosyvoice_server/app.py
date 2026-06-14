"""CosyVoice2-0.5B streaming TTS server (FastAPI).

POST /tts {text, voice, sample_rate, stream} -> streams raw 16-bit PCM mono.
The CosyVoiceTTSService client (local_services/cosyvoice_tts.py) consumes this.

Setup:
    git clone https://github.com/FunAudioLLM/CosyVoice
    # install its requirements, then download the model:
    #   modelscope download iic/CosyVoice2-0.5B --local_dir pretrained_models/CosyVoice2-0.5B
    pip install fastapi uvicorn
    python -m local_services.cosyvoice_server.app   # serves on :8001

VRAM: ~2 GB. Native sample rate is 24 kHz.
"""
from __future__ import annotations

import io
import os

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

MODEL_DIR = os.getenv("COSYVOICE_MODEL_DIR", "pretrained_models/CosyVoice2-0.5B")
DEFAULT_SR = 24000

app = FastAPI(title="CosyVoice2 TTS")
_model = None


def get_model():
    global _model
    if _model is None:
        # Import here so the module loads even before CosyVoice is installed.
        from cosyvoice.cli.cosyvoice import CosyVoice2

        logger.info(f"Loading CosyVoice2 from {MODEL_DIR} …")
        _model = CosyVoice2(MODEL_DIR, load_jit=False, load_trt=False, fp16=True)
        logger.info("CosyVoice2 ready.")
    return _model


class TTSRequest(BaseModel):
    text: str
    voice: str = "default"
    sample_rate: int = DEFAULT_SR
    stream: bool = True


def _to_pcm16(wav: "torch.Tensor", src_sr: int, dst_sr: int) -> bytes:
    """Tensor float [-1,1] -> 16-bit PCM bytes, resampled if needed."""
    audio = wav.squeeze().detach().cpu().float()
    if src_sr != dst_sr:
        import torchaudio

        audio = torchaudio.functional.resample(audio, src_sr, dst_sr)
    pcm = (audio.clamp(-1, 1).numpy() * 32767.0).astype(np.int16)
    return pcm.tobytes()


@app.post("/tts")
def tts(req: TTSRequest):
    model = get_model()

    def gen():
        # CosyVoice2 streaming inference yields speech chunks as they synthesize.
        # `inference_sft` uses a preset speaker; swap for inference_zero_shot to
        # use a cloned voice (pass a prompt wav).
        for out in model.inference_sft(req.text, req.voice, stream=req.stream):
            yield _to_pcm16(out["tts_speech"], model.sample_rate, req.sample_rate)

    return StreamingResponse(gen(), media_type="audio/L16")


@app.get("/health")
def health():
    return {"ok": True, "model_dir": MODEL_DIR}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
