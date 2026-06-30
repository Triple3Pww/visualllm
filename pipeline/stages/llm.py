"""LLM stage factory. Two deliberate, single-provider branches (a fallback switch,
not multi-provider branching), chosen by LLM_PROVIDER:
  weather_chain -> WeatherChainLLMService (dedicated zh weather bot, remote LangServe)
  openrouter    -> OpenAILLMService against OpenRouter (general chat)
Tokens stream so the first sentence reaches TTS before the full answer is done."""
from __future__ import annotations

from pipeline.config import Config


def build_llm(cfg: Config, memory=None):
    if cfg.llm_provider == "weather_chain":
        # Local import keeps this branch off the OpenRouter path (and preflight clean).
        from local_services.weather_chain_llm import WeatherChainLLMService

        return WeatherChainLLMService(
            url=cfg.weather_chain_url,
            model=cfg.weather_chain_model,
            memory=memory,
            verify_tls=cfg.weather_chain_verify_tls,
        )

    from pipecat.services.openai.llm import OpenAILLMService

    # `extra` is merged into the chat.completions.create() call (base_llm.py). For local
    # reasoning models (qwen3.5:4b via Ollama /v1) we must pass reasoning_effort="none" or
    # the model spends ~25-33s thinking before answering. Empty knob -> send nothing, so the
    # cloud-gemini path is byte-for-byte unchanged.
    extra = {}
    if cfg.openrouter_reasoning_effort:
        extra["reasoning_effort"] = cfg.openrouter_reasoning_effort

    # `model=` is deprecated; the model now lives in the `settings=` object.
    settings_kw = dict(model=cfg.openrouter_model, extra=extra)
    if cfg.openrouter_max_tokens:
        # Hard length cap — the safety net for models that ignore the brevity prompt.
        settings_kw["max_completion_tokens"] = cfg.openrouter_max_tokens
    return OpenAILLMService(
        api_key=cfg.openrouter_api_key,
        base_url=cfg.openrouter_base_url,
        settings=OpenAILLMService.Settings(**settings_kw),
    )
