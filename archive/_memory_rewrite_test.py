"""LIVE: build_query rewrites a follow-up via local qwen, and degrades to raw when
the LLM is unreachable. Requires Ollama running with qwen2.5:3b-cpu.
Run: python -m archive._memory_rewrite_test"""
import asyncio
import json
import tempfile
from pathlib import Path

from local_services.avatar_memory import MemoryStore


async def run() -> None:
    d = tempfile.mkdtemp()
    m = MemoryStore(
        base_dir=d, enabled=True,
        llm_url="http://localhost:11434/v1", llm_model="qwen2.5:3b-cpu", gated=True,
    )
    m.profile["default_city"] = "台北市"  # Taipei
    m.record_turn("明天台北市會下雨嗎？", "會的")  # prior: rain tomorrow Taipei
    # follow-up should be rewritten into a self-contained zh question mentioning 台中
    out = await m.build_query("那台中呢？")
    assert "台中" in out, f"expected 台中 in rewrite, got: {len(out)} chars"
    assert "天" in out or "雨" in out, "rewrite lost the weather topic"
    await m.aclose()

    # degradation: bad URL -> returns raw, never raises
    bad = MemoryStore(base_dir=tempfile.mkdtemp(), enabled=True,
                      llm_url="http://127.0.0.1:1/v1", llm_model="x", gated=False)
    raw = "明天會下雨嗎？"
    assert await bad.build_query(raw) == raw
    await bad.aclose()

    # startup recovery: a leftover session.jsonl is distilled + cleared, never raises
    d3 = tempfile.mkdtemp()
    Path(d3, "session.jsonl").write_text(
        json.dumps({"user": "我住在台南市", "bot": "好的", "ts": 0}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    m3 = MemoryStore(base_dir=d3, enabled=True,
                     llm_url="http://localhost:11434/v1", llm_model="qwen2.5:3b-cpu")
    await m3.distill_pending()
    assert Path(d3, "session.jsonl").read_text(encoding="utf-8") == ""  # cleared
    await m3.aclose()
    print("PASS _memory_rewrite_test")


if __name__ == "__main__":
    asyncio.run(run())
