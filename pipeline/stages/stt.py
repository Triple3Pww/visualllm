"""Speech-to-text factory.

Providers:
- deepgram      : streaming API, great for the English prototype.
- azure         : strong zh-TW + Asia-region servers (low latency from Taiwan).
- whisper_local : faster-whisper on the 5060 Ti (en or zh).
- funasr        : Paraformer streaming, strongest Mandarin option (Phase 2/3).
"""
from __future__ import annotations

from pipeline.config import Config


def build_stt(cfg: Config):
    provider = cfg.stt_provider.lower()

    if provider == "deepgram":
        # Pipecat 1.x ships its own LiveOptions compat wrapper.
        from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions

        # Deepgram nova models: use a zh model when in Mandarin mode.
        language = "zh-TW" if cfg.is_mandarin else "en-US"
        return DeepgramSTTService(
            api_key=cfg.deepgram_api_key,
            live_options=LiveOptions(
                model="nova-2-general",
                language=language,
                smart_format=True,
            ),
        )

    if provider == "azure":
        from pipecat.services.azure.stt import AzureSTTService
        from pipecat.transcriptions.language import Language

        # If an enum name differs in your Pipecat version, adjust here.
        language = Language.ZH_TW if cfg.is_mandarin else Language.EN_US
        return AzureSTTService(
            api_key=cfg.azure_speech_key,
            region=cfg.azure_speech_region,
            language=language,
        )

    if provider == "whisper_local":
        # Local CTranslate2 Whisper. zh uses large-v3 for quality; en can use a
        # distilled model for speed.
        from pipecat.services.whisper.stt import WhisperSTTService

        model = "large-v3" if cfg.is_mandarin else "distil-large-v3"
        return WhisperSTTService(model=model, device="cuda")

    if provider == "funasr":
        # Custom wrapper around FunASR Paraformer-streaming (built in Phase 2).
        from local_services.funasr_stt import FunASRSTTService

        return FunASRSTTService(language="zh")

    raise ValueError(f"Unknown STT_PROVIDER: {cfg.stt_provider}")
