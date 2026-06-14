"""LLM factory. Streams tokens; Pipecat aggregates them into sentences so the
first sentence reaches TTS before the full answer is generated.

Providers:
- openai     : GPT-4o-mini, very low TTFT — default prototype LLM.
- anthropic  : Claude Haiku.
- openrouter : one key -> any model (GPT/Claude/Gemini/Qwen/DeepSeek/GLM…).
               Set OPENROUTER_MODEL to pick; great for A/B-testing the LLM,
               especially for the Mandarin target.
- qwen_local : Qwen2.5-7B via an OpenAI-compatible local server (vLLM/Ollama).
               Strongest open Chinese model for the Mandarin target.
"""
from __future__ import annotations

from pipeline.config import Config


def build_llm(cfg: Config):
    provider = cfg.llm_provider.lower()

    if provider == "openai":
        from pipecat.services.openai.llm import OpenAILLMService

        return OpenAILLMService(api_key=cfg.openai_api_key, model="gpt-4o-mini")

    if provider == "anthropic":
        from pipecat.services.anthropic.llm import AnthropicLLMService

        return AnthropicLLMService(
            api_key=cfg.anthropic_api_key, model="claude-haiku-4-5-20251001"
        )

    if provider == "openrouter":
        # OpenRouter is OpenAI-compatible — same service, different base_url.
        from pipecat.services.openai.llm import OpenAILLMService

        return OpenAILLMService(
            api_key=cfg.openrouter_api_key,
            base_url=cfg.openrouter_base_url,
            model=cfg.openrouter_model,
        )

    if provider == "qwen_local":
        # OpenAI-compatible endpoint (vLLM `--served-model-name`, or Ollama).
        from pipecat.services.openai.llm import OpenAILLMService

        return OpenAILLMService(
            api_key="local",
            base_url=cfg.qwen_base_url,
            model="qwen2.5-7b-instruct",
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {cfg.llm_provider}")
