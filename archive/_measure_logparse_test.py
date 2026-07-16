"""logparse anchor extraction from a synthetic pipeline.log snippet.
Run: pytest archive/_measure_logparse_test.py -v"""
from datetime import datetime

from scripts.measure.logparse import (
    build_turn, find_turn_indices, interrupted_turns, parse_beacon_full, parse_playout_beacon,
    reply_seconds, select_driven_turns,
)


def test_select_driven_turns_drops_turns_from_an_earlier_session():
    """The window must not silently borrow turns this run never drove.

    find_turn_indices scans the WHOLE log, so `idxs[-n:]` backfills from earlier sessions
    whenever the driver registers fewer than N [TTFO] lines -- reporting stale, usually WORSE
    turns as a fresh measurement. Measured 2026-07-16: a driver that produced 4 turns had a
    5th backfilled from a pipeline process 2 minutes older; its render (2.20s vs the run's
    0.51s) was read as a cold start that never happened.
    """
    drive_start = 1000.0
    turns = [dict(t0_epoch=900.0, tag="stray"),      # a previous session -> must be dropped
             dict(t0_epoch=1001.0, tag="a"),
             dict(t0_epoch=1002.0, tag="b")]
    got = select_driven_turns(turns, drive_start, want=5)
    assert [t["tag"] for t in got] == ["a", "b"]     # 2 real turns, NOT padded with the stray


def test_select_driven_turns_keeps_everything_when_observing():
    """--observe has no driver: the pre-existing turns ARE the subject, so nothing is dropped."""
    turns = [dict(t0_epoch=900.0, tag="x"), dict(t0_epoch=901.0, tag="y")]
    assert [t["tag"] for t in select_driven_turns(turns, None, want=5)] == ["x", "y"]


def test_select_driven_turns_returns_only_the_newest_want():
    turns = [dict(t0_epoch=float(1000 + i), tag=str(i)) for i in range(6)]
    assert [t["tag"] for t in select_driven_turns(turns, 999.0, want=2)] == ["4", "5"]


def test_interrupted_turns_are_those_whose_reply_never_finished():
    """A turn that speaks to completion logs a bot-stop; one the NEXT turn cuts off never does.

    Overlapped turns do not measure the system -- the interrupt starves the renderer. Measured
    2026-07-16: render 0.5s -> 2.17s under overlap, which read exactly like the documented
    session-degradation bug and was not it.
    """
    turns = [dict(bot_started=2.0, bot_stopped=None),   # cut off by the turn after it
             dict(bot_started=2.0, bot_stopped=50.0),
             dict(bot_started=2.0, bot_stopped=None)]   # LAST turn: see below
    assert interrupted_turns(turns) == [turns[0]]


def test_last_turn_without_a_bot_stop_is_not_called_interrupted():
    """The final turn routinely lacks a bot-stop because the DRIVER disconnected mid-reply.

    Nothing came after it, so nothing interrupted it -- only a turn with a successor can be
    judged. Flagging it would fire the overlap warning on every single run and train the reader
    to ignore it, which is worse than not warning at all.
    """
    assert interrupted_turns([dict(bot_started=2.0, bot_stopped=None)]) == []
    assert interrupted_turns([]) == []


def test_reply_seconds_is_the_longest_completed_reply():
    turns = [dict(bot_started=2.0, bot_stopped=52.0),   # 50s -- the real reply length here
             dict(bot_started=2.0, bot_stopped=12.0),
             dict(bot_started=2.0, bot_stopped=None)]   # interrupted -> not a reply length
    assert reply_seconds(turns) == 50.0
    assert reply_seconds([dict(bot_started=2.0, bot_stopped=None)]) is None


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
