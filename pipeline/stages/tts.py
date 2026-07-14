"""Text-to-speech.

Default: CosyVoice (TTS_PROVIDER=cosyvoice) -- a local CosyVoice2-0.5B streaming
server (female zero-shot voice), no per-token cloud cost; streams first chunk early
enough to feed the avatar within the TTFO target.

Thai (TTS_PROVIDER=jaitts): CosyVoice CANNOT speak Thai, so LANGUAGE=th needs the local
JaiTTS-F5TTS server (local_services/jaitts_server/app.py, port 8004). Same /tts/stream
raw-PCM contract, so it reuses the CosyVoice client.

Fallbacks: ElevenLabs streaming (flash_v2_5, multilingual cloud, covers zh-TW) and
Deepgram Aura (TTS_PROVIDER=deepgram, reuses the Deepgram key, English-only). These
are deliberate fallback switches, not a return to multi-provider branching.
"""
from __future__ import annotations

from pipeline.config import Config


def build_tts(cfg: Config):
    if cfg.tts_provider == "cosyvoice":
        from local_services.cosyvoice_tts import CosyVoiceTTSService

        # Local streaming server; native 24 kHz (Pipecat resamples to 16 kHz for the
        # avatar). No `voice`: the server ignores it and uses its one registered reference
        # (COSYVOICE_PROMPT_WAV/TEXT, swapped via the avatar-preset system).
        return CosyVoiceTTSService(
            base_url=cfg.cosyvoice_url,
            sample_rate=cfg.cosyvoice_sample_rate,
        )

    if cfg.tts_provider == "moss":
        # MOSS-TTS-Realtime local server speaks the SAME /tts/stream raw-PCM contract as
        # CosyVoice, so we reuse the CosyVoice client pointed at MOSS_URL. Its voice is a
        # fixed reference pinned server-side (MOSS_REF); the request `voice` field is ignored.
        # Native 24 kHz; Pipecat resamples to 16 kHz for the avatar.
        from local_services.cosyvoice_tts import CosyVoiceTTSService

        return CosyVoiceTTSService(
            base_url=cfg.moss_url,
            sample_rate=cfg.moss_sample_rate,
        )

    if cfg.tts_provider == "jaitts":
        # JaiTTS-F5TTS local Thai server -- THE Thai voice path (CosyVoice cannot speak
        # Thai). Speaks the SAME /tts/stream raw-PCM contract, so we reuse the CosyVoice
        # client pointed at JAITTS_URL. Voice = a fixed reference clip pinned server-side
        # (JAITTS_REF); the request `voice` field is ignored. Native 24 kHz.
        from local_services.cosyvoice_tts import CosyVoiceTTSService

        return CosyVoiceTTSService(
            base_url=cfg.jaitts_url,
            sample_rate=cfg.jaitts_sample_rate,
        )

    if cfg.tts_provider == "deepgram":
        from pipecat.services.deepgram.tts import DeepgramTTSService

        # Reuses DEEPGRAM_API_KEY (same account as STT). Aura outputs linear16; the
        # avatar/transport resample as needed, so no sample_rate pinning required.
        return DeepgramTTSService(
            api_key=cfg.deepgram_api_key,
            settings=DeepgramTTSService.Settings(voice=cfg.deepgram_tts_voice),
        )

    from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

    # `voice_id=`/`model=` are deprecated; both now live in `settings=`
    # (note the field is `voice`, not `voice_id`).
    return ElevenLabsTTSService(
        api_key=cfg.elevenlabs_api_key,
        settings=ElevenLabsTTSService.Settings(
            voice=cfg.elevenlabs_voice_id,
            model=cfg.elevenlabs_model,
        ),
    )
