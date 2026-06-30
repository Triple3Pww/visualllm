"""Speech-to-text. Default: Deepgram streaming (nova-2), switching model language by
config so one provider serves the English prototype, the zh-TW target, and the Thai
(th) live-character validation. Fallback: STT_PROVIDER=funasr -> a local OFFLINE
SenseVoice-Small server (CPU, ~0 VRAM) for a fully-local zh-TW stack. Deliberate
fallback switch, not multi-provider branching."""
from __future__ import annotations

from pipeline.config import Config


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
    # Pipecat moved per-service tuning into a `settings=` object; the old
    # `live_options=LiveOptions(...)` is deprecated and slated for removal.
    return DeepgramSTTService(
        api_key=cfg.deepgram_api_key,
        settings=DeepgramSTTService.Settings(
            model="nova-2-general",
            language=language,
            smart_format=True,
        ),
    )
