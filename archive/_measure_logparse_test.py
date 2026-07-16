"""logparse anchor extraction from a synthetic pipeline.log snippet.
Run: pytest archive/_measure_logparse_test.py -v"""
from datetime import datetime

from scripts.measure.logparse import (
    build_turn, find_turn_indices, parse_beacon_full, parse_playout_beacon,
)


def _mk(lines_text):
    """('12:00:00.000', 'msg') tuples -> [(datetime, 'ts | msg')] like parse_lines yields."""
    base = "2026-07-16 "
    out = []
    for ts, msg in lines_text:
        dt = datetime.strptime(base + ts, "%Y-%m-%d %H:%M:%S.%f")
        out.append((dt, f"{base + ts} | INFO | {msg}"))
    return out


def test_build_turn_extracts_core_anchors():
    lines = _mk([
        ("12:00:00.000", "Transcription: User stopped speaking"),
        ("12:00:00.010", "Generating chat from context [{'role': 'user', 'content': 'what is ai'}]"),
        ("12:00:00.700", "OpenAILLMService#0 TTFB: 0.690s"),
        ("12:00:00.760", "run_tts:84 - CosyVoice TTS [AI is smart.]"),
        ("12:00:02.450", "CosyVoiceTTSService#0 TTFB: 1.690s"),
        ("12:00:02.600", "[render] first-frame +150ms"),
        ("12:00:02.750", "[TTFO OK ] 2.75s (target 3.0s)"),
        ("12:00:20.000", "Bot stopped speaking based on TTSStoppedFrame"),
    ])
    bi = find_turn_indices(lines)[-1]
    turn = build_turn(lines, bi)
    assert turn["question"] == "what is ai"
    assert abs(turn["ttfo_s"] - 2.75) < 1e-6 and turn["ttfo_pass"] is True
    assert abs(turn["llm_recv"] - 0.010) < 1e-3       # t0 -> 'Generating chat'
    assert abs(turn["llm_ttfb"][0] - 0.700) < 1e-3
    assert abs(turn["tts_recv"] - 0.760) < 1e-3
    assert abs(turn["tts_ttfb"][0][0] - 2.450) < 1e-3
    assert abs(turn["render"] - 2.600) < 1e-3         # [render] line, t0-relative
    assert abs(turn["bot_started"] - 2.750) < 1e-3


def test_parse_playout_beacon_offset_old_and_new_shapes():
    t0 = datetime.fromtimestamp(1751800000.0)
    old = [(datetime.fromtimestamp(1751800000.5),
            '2026 | INFO | [client-playout] {"ev":"audio-onset","t":1751800000123}')]
    new = [(datetime.fromtimestamp(1751800000.5),
            '2026 | INFO | [client-playout] {"onset":1751800000200,"recv":1751800000050,"jitterMs":90,"rttMs":20}')]
    assert abs(parse_playout_beacon(old, t0) - 0.123) < 1e-6
    assert abs(parse_playout_beacon(new, t0) - 0.200) < 1e-6


def test_parse_beacon_full_returns_all_fields_t0_relative():
    t0 = datetime.fromtimestamp(1751800000.0)
    lines = [(datetime.fromtimestamp(1751800000.5),
              '2026 | INFO | [client-playout] {"onset":1751800000200,"recv":1751800000050,"jitterMs":90,"rttMs":20}')]
    b = parse_beacon_full(lines, t0)
    assert abs(b["recv"] - 0.050) < 1e-6
    assert abs(b["onset"] - 0.200) < 1e-6
    assert abs(b["jitter"] - 0.090) < 1e-6
    assert abs(b["rtt"] - 0.020) < 1e-6
