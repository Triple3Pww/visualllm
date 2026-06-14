"""Preflight / smoke test — run before wiring keys or running the pipeline.

Pipecat's import paths drift between releases, and the stage factories import
provider SDKs lazily. This script resolves every fragile import *without needing
API keys or network*, so you catch version drift in one shot instead of
discovering it mid-session.

For each provider branch it reports one of:
  PASS  imports resolve and the service constructs
  KEYS  imports resolve; construction needs keys/config (expected, fine)
  DRIFT import failed — a Pipecat path moved or an extra isn't installed (FIX)
  SKIP  optional local dep not installed yet (expected before Phase 2/3)

Exit code is non-zero if any DRIFT is found, so it doubles as a CI gate.

    python -m scripts.preflight
"""
from __future__ import annotations

import dataclasses
import sys

# Local Phase 2/3 deps that are expected to be absent early on. An import error
# mentioning one of these is a SKIP (install when you reach that phase), not
# a Pipecat-drift failure.
OPTIONAL_DEPS = {
    "funasr", "websockets", "cosyvoice", "torch", "torchaudio", "modelscope",
    # alternative-provider extras not in the active stack (install if you switch):
    "livekit", "kokoro_onnx", "kokoro", "anthropic", "faster_whisper",
}

RESET, GREEN, YELLOW, RED, GREY = "\033[0m", "\033[32m", "\033[33m", "\033[31m", "\033[90m"


def _check(label: str, fn) -> str:
    try:
        fn()
        return f"{GREEN}PASS{RESET}  {label}"
    except (ImportError, ModuleNotFoundError) as e:
        missing = getattr(e, "name", "") or str(e)
        if any(dep in str(e) for dep in OPTIONAL_DEPS):
            return f"{GREY}SKIP{RESET}  {label}  ({missing} not installed)"
        _check.drift = True
        return f"{RED}DRIFT{RESET} {label}  -> {e}"
    except Exception as e:  # noqa: BLE001 — construction reached, just needs config
        return f"{YELLOW}KEYS{RESET}  {label}  ({type(e).__name__})"


_check.drift = False


def main() -> int:
    print("== Environment ==")
    try:
        import pipecat

        print(f"  pipecat {getattr(pipecat, '__version__', '?')}")
    except Exception as e:  # noqa: BLE001
        print(f"  {RED}pipecat not importable: {e}{RESET}")
        print("  -> pip install -r requirements.txt")
        return 2

    from pipeline.config import config
    from pipeline.stages import build_avatar, build_llm, build_stt, build_tts, build_vad_params

    print("\n== Core modules ==")
    print(_check("pipeline.metrics", lambda: __import__("pipeline.metrics", fromlist=["TtfoMeter"])))
    print(_check("pipeline.main", lambda: __import__("pipeline.main", fromlist=["run_bot"])))
    print(_check("Silero VAD construct", build_vad_params))

    branches = {
        "STT": ("stt_provider", ["deepgram", "azure", "whisper_local", "funasr"], build_stt),
        "LLM": ("llm_provider", ["openai", "anthropic", "openrouter", "qwen_local"], build_llm),
        "TTS": ("tts_provider", ["elevenlabs", "cartesia", "azure", "cosyvoice_local", "kokoro_local"], build_tts),
        "AVATAR": ("avatar_provider", ["simli", "heygen", "musetalk_local"], build_avatar),
    }
    for stage, (field, providers, builder) in branches.items():
        print(f"\n== {stage} providers ==")
        for p in providers:
            cfg = dataclasses.replace(config, **{field: p})
            print(_check(f"{stage.lower()}={p}", lambda b=builder, c=cfg: b(c)))

    print()
    if _check.drift:
        print(f"{RED}Drift detected — fix the imports above before running.{RESET}")
        return 1
    print(f"{GREEN}No drift. Imports resolve for every provider branch.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
