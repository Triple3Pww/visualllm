"""Speech-to-text. Default: Deepgram streaming (nova-2), switching model language by
config so one provider serves the English prototype, the zh-TW target, and the Thai
(th) live-character validation.

Local fallback (fully offline, CPU, ~0 VRAM):
  STT_PROVIDER=sherpa -> sherpa-onnx STREAMING zipformer (bilingual zh-en); drives
    turn-taking from its own ASR endpoint detector, robust to a quiet/attenuated mic.

(Removed 2026-07-14: the funasr branch. STT_PROVIDER=funasr was never selected and the
project's own docs called it an "untested alt"; it is in git history if it is ever wanted.)
"""
from __future__ import annotations

from pipeline.config import Config


def build_stt(cfg: Config):
    if cfg.stt_provider == "sherpa":
        # Local OFFLINE STREAMING (sherpa-onnx, CPU, ~0 VRAM). Drives turn-taking from its
        # own ASR endpoint detector, so it works even when the energy-VAD doesn't fire.
        from local_services.sherpa_stt import SherpaStreamingSTTService

        return SherpaStreamingSTTService(
            model_dir=cfg.sherpa_model_dir,
            to_traditional=cfg.sherpa_traditional,
            endpoint_silence=cfg.sherpa_endpoint_silence,
            # Only pause on bot speech when echo-guard is on (valid only with live sync). Under the
            # default steady sync the resume signal never fires, so pausing would strand the mic
            # after the greeting (P11); default OFF keeps the mic live (barge-in/headphones).
            pause_while_bot_speaks=cfg.echo_guard,
        )

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
