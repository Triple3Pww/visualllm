"""Pure latency functions for the measure waterfall -- onset, waterfall sum, beacon parse.
Run: python -m archive._measure_waterfall_test  (or: pytest archive/_measure_waterfall_test.py)"""
from datetime import datetime

from scripts.measure import (
    aggregate_turns, answer_onset_epoch, build_waterfall, parse_playout_beacon,
)


def test_onset_ignores_greeting_and_silence():
    # greeting burst BEFORE t0, then silence, then the real answer after t0.
    t0 = 1000.0
    samples = (
        [(998.0 + i * 0.02, 0.5) for i in range(10)]   # greeting, pre-t0 -> ignored
        + [(1000.0 + i * 0.02, 0.0) for i in range(20)]  # silence after t0 -> below thresh
        + [(1000.4 + i * 0.02, 0.6) for i in range(20)]  # answer -> onset here
    )
    onset = answer_onset_epoch(samples, t0)
    assert onset is not None
    assert abs(onset - 1000.4) < 1e-6


def test_onset_needs_a_sustained_run_not_a_single_spike():
    t0 = 0.0
    samples = [(0.2, 0.9)] + [(0.2 + i * 0.02, 0.0) for i in range(1, 10)] \
        + [(0.6 + i * 0.02, 0.9) for i in range(5)]
    onset = answer_onset_epoch(samples, t0, run=3)
    assert abs(onset - 0.6) < 1e-6  # the lone 0.2s spike is skipped


def test_onset_all_silence_returns_none():
    assert answer_onset_epoch([(1.0, 0.0), (1.1, 0.0), (1.2, 0.0)], 0.0) is None


def test_waterfall_deltas_telescope_to_total():
    anchors = dict(llm_recv=0.0, llm_ttfb=0.68, tts_recv=1.05, tts_ttfb=2.45,
                   bot_started=2.75, client_arrival=2.97, playout=3.12)
    rows = build_waterfall(anchors, playout_source="browser")
    ok = [r for r in rows if r["status"] == "ok"]
    total = [r for r in rows if r["status"] == "total"][0]["cum"]
    assert abs(sum(r["delta"] for r in ok) - total) < 1e-6
    assert abs(total - 3.12) < 1e-6
    assert ok[-1]["source"] == "browser"


def test_waterfall_missing_anchor_is_unknown_and_sum_still_holds():
    anchors = dict(llm_recv=0.0, llm_ttfb=0.68, tts_recv=1.05, tts_ttfb=None,
                   bot_started=2.75, client_arrival=2.97, playout=None)
    rows = build_waterfall(anchors)
    by_stage = {r["stage"]: r for r in rows}
    assert by_stage["TTS synth first chunk"]["status"] == "unknown"
    ok = [r for r in rows if r["status"] == "ok"]
    total = [r for r in rows if r["status"] == "total"][0]["cum"]
    assert abs(sum(r["delta"] for r in ok) - total) < 1e-6  # telescoping survives the gap
    assert abs(total - 2.97) < 1e-6  # last known cum (playout missing -> est. fills in caller)


def test_waterfall_negative_anchor_is_unknown_not_negative_latency():
    # A prior-turn artifact: llm_ttfb logged 1.34s BEFORE t0 (VAD split a synthetic mic).
    # It must render 'unknown', never a physically-impossible -1.34s latency, and the sum
    # must still telescope through the later real anchors.
    anchors = dict(llm_recv=0.0, llm_ttfb=-1.34, tts_recv=0.64, tts_ttfb=1.73,
                   bot_started=2.26, client_arrival=3.06, playout=3.46)
    rows = build_waterfall(anchors, playout_source="est")
    by_stage = {r["stage"]: r for r in rows}
    assert by_stage["LLM first token"]["status"] == "unknown"
    assert by_stage["LLM first token"]["delta"] is None  # no negative latency shown
    ok = [r for r in rows if r["status"] == "ok"]
    assert all(r["delta"] >= 0 for r in ok)  # every shown delta is non-negative
    total = [r for r in rows if r["status"] == "total"][0]["cum"]
    assert abs(sum(r["delta"] for r in ok) - total) < 1e-6
    assert abs(total - 3.46) < 1e-6


def test_parse_playout_beacon_offset():
    t0 = datetime.fromtimestamp(1751800000.0)
    lines = [
        (datetime.fromtimestamp(1751799999.0), "something before t0 audio-onset t=1"),
        (datetime.fromtimestamp(1751800000.5), '[client-playout] {"ev":"audio-onset","t":1751800000123}'),
    ]
    assert parse_playout_beacon(lines, t0) == 0.123


def test_capture_offsets_every_downstream_cum():
    anchors = dict(llm_recv=0.0, llm_ttfb=0.68, tts_recv=1.05, tts_ttfb=2.45,
                   render=2.60, bot_started=2.75, client_arrival=2.97, jitter=3.05, playout=3.12)
    rows = build_waterfall(anchors, playout_source="browser-audio", capture=0.70)
    cap = [r for r in rows if r["stage"].startswith("Capture")][0]
    assert abs(cap["delta"] - 0.70) < 1e-6 and abs(cap["cum"] - 0.70) < 1e-6
    total = [r for r in rows if r["status"] == "total"][0]["cum"]
    assert abs(total - (3.12 + 0.70)) < 1e-6           # capture shifts the whole sum
    ok = [r for r in rows if r["status"] == "ok"]
    assert abs(sum(r["delta"] for r in ok) - total) < 1e-6


def test_capture_absent_is_unknown_and_keeps_t0_relative_total():
    anchors = dict(llm_recv=0.0, llm_ttfb=0.68, tts_recv=1.05, tts_ttfb=2.45,
                   bot_started=2.75, client_arrival=2.97, playout=3.12)
    rows = build_waterfall(anchors, playout_source="browser", capture=None)
    cap = [r for r in rows if r["stage"].startswith("Capture")][0]
    assert cap["status"] == "unknown" and cap["delta"] is None
    total = [r for r in rows if r["status"] == "total"][0]["cum"]
    assert abs(total - 3.12) < 1e-6                     # no capture -> t0-relative, unchanged


def test_new_stages_carry_levers_and_sources():
    rows = build_waterfall(dict(llm_recv=0.0, render=2.6, jitter=3.05, playout=3.12))
    by = {r["stage"]: r for r in rows}
    assert "lever" in by["Avatar render (first frame)"]
    assert by["Browser jitter buffer"]["source"] == "browser-stats"


def test_aggregate_median_p95_and_fresh_warm():
    turns = [dict(llm_ttfb=t) for t in [0.6, 0.7, 0.8, 0.9, 1.6, 1.7]]
    agg = aggregate_turns(turns, ["llm_ttfb"])
    assert abs(agg["median"]["llm_ttfb"] - 0.85) < 1e-6   # median of 6 = mean of 0.8,0.9
    assert agg["p95"]["llm_ttfb"] >= 1.6
    assert abs(agg["fresh"]["llm_ttfb"] - 0.65) < 1e-6    # turns 0-1 median
    assert abs(agg["warm"]["llm_ttfb"] - 1.65) < 1e-6     # turns >=4 median


def test_aggregate_ignores_none_anchors():
    turns = [dict(tts_ttfb=None), dict(tts_ttfb=2.0), dict(tts_ttfb=2.4)]
    agg = aggregate_turns(turns, ["tts_ttfb"])
    assert abs(agg["median"]["tts_ttfb"] - 2.2) < 1e-6    # None dropped


def test_history_append_and_read(tmp_path, monkeypatch):
    import scripts.measure.report as rep
    monkeypatch.setattr(rep, "HISTORY", tmp_path / "h.jsonl")
    rep.append_history({"meta": {"when": "2026-07-16 12:00", "turns": 5, "e2e_median": 2.9,
                                 "stage_medians": {"LLM first token": 0.7}}}, {"LANGUAGE": "zh"})
    rows = rep.read_history()
    assert rows[-1]["e2e_median"] == 2.9 and rows[-1]["env"]["LANGUAGE"] == "zh"


def main():
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print("PASS _measure_waterfall_test")


if __name__ == "__main__":
    main()
