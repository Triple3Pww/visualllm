"""Latency profiling helpers.

Two layers of measurement:

1. End-to-end TTFO (the acceptance metric) is collected live by
   pipeline/metrics.py::TtfoMeter during a real session, and printed in
   TtfoMeter.summary() on disconnect. That is the number that must be < 8s p95.

2. Per-stage micro-benchmarks (below) help you find which stage dominates, so
   you know what to tune. They time the two stages that usually matter most:
   LLM time-to-first-token and TTS time-to-first-audio-chunk.

Usage:
    python -m scripts.bench_latency --stage llm  --text "Tell me about Taiwan."
    python -m scripts.bench_latency --stage tts  --text "Hello, how are you?"
"""
from __future__ import annotations

import argparse
import asyncio
import time

from loguru import logger

from pipeline.config import config


def _llm_client_and_model() -> tuple["AsyncOpenAI", str]:
    """Resolve (client, model) for the configured LLM provider.

    Pipecat services run inside a pipeline, not standalone, so the cleanest TTFT
    probe calls the underlying SDK directly. We mirror the same provider/model
    selection as pipeline/stages/llm.py so the benchmark measures the stack that
    actually runs (OpenAI-compatible providers all use AsyncOpenAI + a base_url).
    """
    from openai import AsyncOpenAI

    provider = config.llm_provider.lower()
    if provider == "openai":
        return AsyncOpenAI(api_key=config.openai_api_key), "gpt-4o-mini"
    if provider == "openrouter":
        return (
            AsyncOpenAI(api_key=config.openrouter_api_key,
                        base_url=config.openrouter_base_url),
            config.openrouter_model,
        )
    if provider == "qwen_local":
        return (
            AsyncOpenAI(api_key="local", base_url=config.qwen_base_url),
            "qwen2.5-7b-instruct",
        )
    if provider == "anthropic":
        # Anthropic isn't OpenAI-compatible; bench it via its own SDK below.
        raise NotImplementedError("anthropic")
    raise ValueError(f"Unknown LLM_PROVIDER for bench: {config.llm_provider}")


async def bench_llm(prompt: str) -> float:
    """Time-to-first-token from the configured LLM provider."""
    provider = config.llm_provider.lower()

    if provider == "anthropic":
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=config.anthropic_api_key)
        t0 = time.monotonic()
        async with client.messages.stream(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            system=config.system_prompt,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                if text:
                    ttft = time.monotonic() - t0
                    logger.info(f"LLM time-to-first-token: {ttft:0.3f}s")
                    return ttft
        return float("nan")

    client, model = _llm_client_and_model()
    t0 = time.monotonic()
    stream = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            ttft = time.monotonic() - t0
            logger.info(f"LLM time-to-first-token: {ttft:0.3f}s")
            return ttft
    return float("nan")


async def bench_tts(text: str) -> float:
    """Time-to-first-audio-chunk from the configured TTS provider (ElevenLabs)."""
    import aiohttp

    t0 = time.monotonic()
    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech/{config.elevenlabs_voice_id}"
        "/stream?optimize_streaming_latency=4"
    )
    headers = {"xi-api-key": config.elevenlabs_api_key, "Content-Type": "application/json"}
    body = {"text": text, "model_id": "eleven_flash_v2_5"}
    async with aiohttp.ClientSession() as s:
        async with s.post(url, headers=headers, json=body) as r:
            async for _ in r.content.iter_chunked(1024):
                ttfa = time.monotonic() - t0
                logger.info(f"TTS time-to-first-audio: {ttfa:0.3f}s")
                return ttfa
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["llm", "tts"], required=True)
    ap.add_argument("--text", default="Tell me one interesting fact about Taiwan.")
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    async def go():
        fn = bench_llm if args.stage == "llm" else bench_tts
        if args.stage == "llm":
            logger.info(f"LLM provider={config.llm_provider} model={config.openrouter_model if config.llm_provider.lower()=='openrouter' else ''}")
        samples = [await fn(args.text) for _ in range(args.runs)]
        samples.sort()
        logger.info(
            f"{args.stage} over {args.runs} runs -- "
            f"median={samples[len(samples)//2]:0.3f}s max={samples[-1]:0.3f}s"
        )

    asyncio.run(go())


if __name__ == "__main__":
    main()
