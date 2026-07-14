"""Text-to-speech.

Default: CosyVoice (TTS_PROVIDER=cosyvoice) -- the local CosyVoice streaming server that now
lives in this repo at tts/cosyvoice-server/ (run it on vLLM in WSL). Female zero-shot voice, no
per-token cloud cost; streams the first chunk early enough to feed the avatar within the TTFO
target.

Thai (TTS_PROVIDER=jaitts): CosyVoice CANNOT speak Thai, so LANGUAGE=th needs the local
JaiTTS-F5TTS server (local_services/jaitts_server/app.py, port 8004). It speaks the SAME
/tts/stream raw-PCM contract, so it reuses the CosyVoice client pointed at JAITTS_URL.

(Removed 2026-07-14: the moss / elevenlabs / deepgram branches. They were never selected --
TTS_PROVIDER has been `cosyvoice` throughout -- and an untried fallback is not a safety net,
it is code that rots. `git revert` brings any of them back if a fallback is ever needed.)
"""
from __future__ import annotations

from pipeline.config import Config


def build_tts(cfg: Config):
    if cfg.tts_provider == "jaitts":
        # JaiTTS-F5TTS local Thai server -- THE Thai voice path (CosyVoice cannot speak Thai).
        # Speaks the SAME /tts/stream raw-PCM contract, so we reuse the CosyVoice client pointed
        # at JAITTS_URL. Voice = a fixed reference clip pinned server-side (JAITTS_REF); 24 kHz.
        from local_services.cosyvoice_tts import CosyVoiceTTSService

        return CosyVoiceTTSService(
            base_url=cfg.jaitts_url,
            sample_rate=cfg.jaitts_sample_rate,
        )

    if cfg.tts_provider != "cosyvoice":
        # Fail loudly rather than silently speaking in the wrong engine: a typo'd TTS_PROVIDER
        # used to fall through to ElevenLabs (a CLOUD voice, and a bill), which is a bad default.
        raise ValueError(
            f"TTS_PROVIDER={cfg.tts_provider!r} is not supported. Use 'cosyvoice' (default) "
            f"or 'jaitts' (Thai)."
        )

    from local_services.cosyvoice_tts import CosyVoiceTTSService

    # Local streaming server; native 24 kHz (Pipecat resamples to 16 kHz for the avatar).
    # No `voice`: the server ignores it and uses its one registered reference
    # (COSYVOICE_PROMPT_WAV/TEXT, swapped via the avatar-preset system).
    return CosyVoiceTTSService(
        base_url=cfg.cosyvoice_url,
        sample_rate=cfg.cosyvoice_sample_rate,
    )
