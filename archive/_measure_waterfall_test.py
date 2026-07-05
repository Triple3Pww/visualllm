"""Pure latency functions for the measure waterfall — onset, waterfall sum, beacon parse.
Run: python -m archive._measure_waterfall_test  (or: pytest archive/_measure_waterfall_test.py)"""
from datetime import datetime

from scripts.measure import answer_onset_epoch, build_waterfall, parse_playout_beacon


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


def test_parse_playout_beacon_offset():
    t0 = datetime.fromtimestamp(1751800000.0)
    lines = [
        (datetime.fromtimestamp(1751799999.0), "something before t0 audio-onset t=1"),
        (datetime.fromtimestamp(1751800000.5), '[client-playout] {"ev":"audio-onset","t":1751800000123}'),
    ]
    assert parse_playout_beacon(lines, t0) == 0.123


def main():
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print("PASS _measure_waterfall_test")


if __name__ == "__main__":
    main()
