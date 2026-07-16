"""The beacon endpoint logs exactly one [client-playout] {json} line; logparse must read it
back on the pipeline clock. This locks that log-line contract (the live HTTP path is smoke-
tested in the plan's Task 10). Run: pytest archive/_measure_beacon_test.py -v"""
import json
from datetime import datetime

from scripts.measure.logparse import parse_beacon_full


def test_beacon_json_roundtrips_through_logparse():
    body = {"onset": 1751800000200, "recv": 1751800000050, "jitterMs": 90, "rttMs": 20}
    line = f"2026-07-16 12:00:00.500 | INFO | [client-playout] {json.dumps(body)}"
    t0 = datetime.fromtimestamp(1751800000.0)
    b = parse_beacon_full([(t0, line)], t0)
    assert abs(b["onset"] - 0.200) < 1e-6 and abs(b["jitter"] - 0.090) < 1e-6
    assert abs(b["recv"] - 0.050) < 1e-6 and abs(b["rtt"] - 0.020) < 1e-6
