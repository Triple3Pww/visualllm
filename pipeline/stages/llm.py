"""LLM: OpenRouter (one key -> any model via OPENROUTER_MODEL). OpenRouter is
OpenAI-compatible, so Pipecat's OpenAI service drives it with a different base_url.
Tokens stream so the first sentence reaches TTS before the full answer is done."""
from __future__ import annotations

from pipeline.config import Config


def build_llm(cfg: Config):
    from pipecat.services.openai.llm import OpenAILLMService

    # `model=` is deprecated; the model now lives in the `settings=` object.
    return OpenAILLMService(
        api_key=cfg.openrouter_api_key,
        base_url=cfg.openrouter_base_url,
        settings=OpenAILLMService.Settings(model=cfg.openrouter_model),
    )
