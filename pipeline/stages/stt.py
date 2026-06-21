"""Speech-to-text: Deepgram streaming (nova-2). Switches model language by config
so the same provider serves the English prototype, the zh-TW target, and the Thai
(th) live-character validation."""
from __future__ import annotations

from pipeline.config import Config


def build_stt(cfg: Config):
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
