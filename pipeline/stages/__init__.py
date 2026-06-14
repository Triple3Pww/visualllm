"""Stage factories. Each build_* function returns a Pipecat processor chosen by
config, so pipeline/main.py stays provider-agnostic."""
from .avatar import build_avatar
from .llm import build_llm
from .stt import build_stt
from .tts import build_tts
from .vad import build_vad_params

__all__ = ["build_stt", "build_llm", "build_tts", "build_avatar", "build_vad_params"]
