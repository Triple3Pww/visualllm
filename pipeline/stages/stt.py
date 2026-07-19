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
            # Only pause on bot speech when echo-guard is on. The resume signal it needs
            # (BotStopped) fires correctly under steady since P53 -- live-verified 2026-07-17,
            # so the old "strands the mic after the greeting" blocker (P11) is gone. Default is
            # still OFF: that's echo-guard's own posture (barge-in/headphones), not a defect.
            # See local_services/sherpa_stt.py::__init__ for the full why.
            pause_while_bot_speaks=cfg.echo_guard,
        )

    if cfg.stt_provider == "sensevoice":
        # Local OFFLINE SenseVoice-Small (segmented via the pipeline VAD + Smart Turn). Higher
        # accuracy + far more noise-robust than the streaming zipformer, ~0.5GB on the GPU.
        # Turn-taking comes from the transport VAD here, NOT an ASR endpoint detector -- if a
        # quiet mic fails to trip the VAD, fall back to STT_PROVIDER=sherpa (endpoint-driven).
        from local_services.sensevoice_stt import SenseVoiceSTTService

        return SenseVoiceSTTService(
            model_dir=cfg.sensevoice_model_dir,
            provider=cfg.sensevoice_provider,
            to_traditional=cfg.sensevoice_traditional,
            # Reuse the streaming zipformer purely as the endpoint detector (proven turn-taking
            # on this box); SenseVoice does the accurate transcription of each buffered utterance.
            endpoint_model_dir=cfg.sherpa_model_dir,
            endpoint_silence=cfg.sensevoice_endpoint_silence,
            pause_while_bot_speaks=cfg.echo_guard,
        )

    if cfg.stt_provider != "deepgram":
        # Fail loudly rather than silently transcribing on the wrong engine: a typo'd
        # STT_PROVIDER used to fall through to Deepgram (a CLOUD service, and a bill) --
        # the same bad default the TTS factory already refuses.
        raise ValueError(
            f"STT_PROVIDER={cfg.stt_provider!r} is not supported. Use 'deepgram' (default), "
            f"'sherpa' (local streaming) or 'sensevoice' (local offline, GPU)."
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
