"""Central configuration: one place to select providers and read env.

Every stage factory (pipeline/stages/*.py) reads from this Config object, so the
whole pipeline can be re-pointed (English<->Mandarin, API<->local) by editing
.env only — no code changes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str | None = None) -> str | None:
    val = os.getenv(name, default)
    return val.strip() if isinstance(val, str) else val


@dataclass(frozen=True)
class Config:
    # --- stage selection ---
    stt_provider: str = _get("STT_PROVIDER", "deepgram")
    llm_provider: str = _get("LLM_PROVIDER", "openai")
    tts_provider: str = _get("TTS_PROVIDER", "elevenlabs")
    avatar_provider: str = _get("AVATAR_PROVIDER", "simli")
    language: str = _get("LANGUAGE", "en")  # "en" | "zh"

    # --- keys / ids ---
    deepgram_api_key: str | None = _get("DEEPGRAM_API_KEY")
    openai_api_key: str | None = _get("OPENAI_API_KEY")
    anthropic_api_key: str | None = _get("ANTHROPIC_API_KEY")
    openrouter_api_key: str | None = _get("OPENROUTER_API_KEY")
    elevenlabs_api_key: str | None = _get("ELEVENLABS_API_KEY")
    elevenlabs_voice_id: str | None = _get("ELEVENLABS_VOICE_ID")
    simli_api_key: str | None = _get("SIMLI_API_KEY")
    simli_face_id: str | None = _get("SIMLI_FACE_ID")
    heygen_api_key: str | None = _get("HEYGEN_API_KEY")
    heygen_avatar_id: str | None = _get("HEYGEN_AVATAR_ID")
    # Azure (STT + TTS) — strong zh-TW + Asia-region servers (low latency from TW)
    azure_speech_key: str | None = _get("AZURE_SPEECH_KEY")
    azure_speech_region: str | None = _get("AZURE_SPEECH_REGION", "eastasia")
    azure_tts_voice: str | None = _get("AZURE_TTS_VOICE")  # blank -> chosen by language
    # Cartesia (TTS) — very low latency
    cartesia_api_key: str | None = _get("CARTESIA_API_KEY")
    cartesia_voice_id: str | None = _get("CARTESIA_VOICE_ID")

    # --- local endpoints ---
    cosyvoice_url: str = _get("COSYVOICE_URL", "http://localhost:8001")
    musetalk_url: str = _get("MUSETALK_URL", "http://localhost:8002")
    qwen_base_url: str = _get("QWEN_BASE_URL", "http://localhost:8000/v1")
    openrouter_base_url: str = _get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    openrouter_model: str = _get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

    # --- targets ---
    ttfo_target_s: float = float(_get("TTFO_TARGET_SECONDS", "8"))

    @property
    def is_mandarin(self) -> bool:
        return self.language.lower().startswith("zh")

    @property
    def azure_voice(self) -> str:
        """Default Azure neural voice by language unless AZURE_TTS_VOICE is set."""
        if self.azure_tts_voice:
            return self.azure_tts_voice
        return "zh-TW-HsiaoChenNeural" if self.is_mandarin else "en-US-AvaNeural"

    @property
    def system_prompt(self) -> str:
        if self.is_mandarin:
            return (
                "你是一個友善、簡潔的語音助理。"
                "請用口語化、適合朗讀的方式回答，句子要短，"
                "避免使用表情符號、條列符號或特殊格式。"
            )
        return (
            "You are a friendly, concise voice assistant. Answer in a natural, "
            "spoken style. Keep sentences short. Do not use emojis, bullet "
            "points, or any special formatting — your text will be read aloud."
        )


config = Config()
