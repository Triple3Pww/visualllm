# Mic-to-ear latency harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remake the measure system into a detailed mic-to-ear latency harness: an ~11-stage waterfall (each stage tagged with source + `.env` lever), real browser output delay via both an automated Playwright Chromium and a `/studio` human beacon, a measured pre-t0 capture segment, N-turn median/p95 with a fresh-vs-warm split, and run-history trend tracking.

**Architecture:** `scripts/measure.py` becomes a focused package `scripts/measure/` (entry `python -m scripts.measure` unchanged). Pure logic (`waterfall.py`, `logparse.py`) is TDD'd against fixtures. Browser latency is captured by a `[client-playout] {json}` line the server writes to `pipeline.log` (fed by either the Playwright page or the `?measure=1` `/studio` beacon), so both paths land on the pipeline's single clock. Three tiny additive instrumentation points: a `[render]` log line, a beacon endpoint, and the studio beacon JS. `pipeline/metrics.py` (TtfoMeter) is untouched.

**Tech Stack:** Python 3.11 (system env), pipecat 1.3.0, aiortc, Playwright (already installed), pytest 9.1.1, vanilla-JS studio client, `docs/workflow-timeline.html` + `docs/measure_data.js`.

## Global Constraints

- Entry point `python -m scripts.measure` and output paths (`output/measure_report.json`, `docs/measure_data.js`) MUST stay identical — `docs/workflow-timeline.html` reads `measure_data.js` verbatim.
- **Do NOT modify `pipeline/metrics.py` (TtfoMeter)** — it is deliberately untouched; the waterfall is derived in `scripts/measure/`.
- Server-side `.py` files must stay **ASCII-safe** (Windows console is cp1252). Use `->`, `--`, not Unicode arrows/em-dashes in `.py` source. (Markdown/HTML may use Unicode.)
- **loguru is BRACE-style**: use f-strings, never `logger.info("x=%s", v)`.
- **Never do blocking I/O on the pipeline event loop** — the beacon endpoint must be async and non-blocking.
- The avatar server is **single-client**: turns run sequentially; fully close any browser tab before running.
- Preserve the existing archive tests: `from scripts.measure import answer_onset_epoch, build_waterfall, parse_playout_beacon` must keep working, and the 7 existing test cases in `archive/_measure_waterfall_test.py` must stay green.
- Tests live in `archive/_*_test.py` (run with `pytest`), matching repo convention. New tests go there too.
- Comments state the *why* (latency / a pipecat quirk / hardware), matching house voice.

---

## File structure

**New package (replaces `scripts/measure.py`):**
- `scripts/measure/__init__.py` — re-exports the pure funcs the archive test imports.
- `scripts/measure/waterfall.py` — stage table, `build_waterfall`, `answer_onset_epoch`, `aggregate_turns` (median/p95, fresh/warm).
- `scripts/measure/logparse.py` — pipeline.log -> per-turn anchors, `parse_playout_beacon`, `parse_beacon_full`.
- `scripts/measure/drive.py` — `run_probe` (headless aiortc, reuses `_webrtc_probe`) + `run_browser_turn` (Playwright).
- `scripts/measure/report.py` — JSON / `measure_data.js` / `measure_history.jsonl` writers, `compare_runs`, console summary, `build_events`/`build_handoffs`/`build_metrics`.
- `scripts/measure/__main__.py` — CLI + orchestration (drive N turns, stitch clocks, aggregate, assemble, write).

**Instrumentation (additive):**
- `local_services/musetalk_video.py` — one `[render] first-frame +Xms` log line per turn.
- `pipeline/main.py` — `POST /client/measure-beacon` (logs `[client-playout] {json}`) in the existing `_inject_client_patches` middleware, + `measureBeacon`/`serverEpochMs` in `GET /client/ice-config`.
- `local_services/studio_client/index.html` — `?measure=1` beacon block.

**Presentation:**
- `docs/workflow-timeline.html` — lever column, source tags, capture + browser tail, run-history sparkline.

**Tests:**
- `archive/_measure_waterfall_test.py` — extended (keep existing 7, add capture/render/jitter/aggregate).
- `archive/_measure_logparse_test.py` — new, against a fixture log snippet.

---

### Task 1: `waterfall.py` — detailed stage table, telescoping, aggregation

**Files:**
- Create: `scripts/measure/__init__.py`
- Create: `scripts/measure/waterfall.py`
- Test: `archive/_measure_waterfall_test.py` (extend in place)

**Interfaces:**
- Produces:
  - `answer_onset_epoch(samples: list[tuple[float,float]], t0_epoch: float, guard=0.15, thresh_frac=0.18, run=3) -> float | None` (moved verbatim from current `measure.py`).
  - `_WATERFALL_STAGES: list[tuple[str,str,str,str]]` = `(label, anchor_key, source_default, lever)`.
  - `build_waterfall(anchors: dict, playout_source="est", capture: float | None = None) -> list[dict]`. Rows: `{stage, delta, cum, source, lever, status}`; a final `status="total"` row. `capture` (= `t0 - speech_end`, seconds, or None) is prepended as the first row and its value is added to every downstream `cum`.
  - `aggregate_turns(turns: list[dict], keys: list[str]) -> dict` -> `{"median": {k: float|None}, "p95": {k: float|None}, "fresh": {k: ...}, "warm": {k: ...}}`. `fresh` = turns 0-1, `warm` = turns index >=4 (fallback: last 2 if fewer than 5).

- [ ] **Step 1: Write failing tests (extend the file; keep all 7 existing cases untouched, add these)**

Append to `archive/_measure_waterfall_test.py`:

```python
from scripts.measure import aggregate_turns  # add to the existing import line


def test_capture_offsets_every_downstream_cum():
    anchors = dict(llm_recv=0.0, llm_ttfb=0.68, tts_recv=1.05, tts_ttfb=2.45,
                   render=2.60, bot_started=2.75, client_arrival=2.97, jitter=3.05, playout=3.12)
    rows = build_waterfall(anchors, playout_source="browser-audio", capture=0.70)
    by = {r["stage"]: r for r in rows}
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest archive/_measure_waterfall_test.py -v`
Expected: import error / failures for the new cases (module `scripts.measure.waterfall` not present; `aggregate_turns` undefined).

- [ ] **Step 3: Implement `scripts/measure/waterfall.py`**

```python
"""Pure latency math for the mic-to-ear waterfall: onset detection, the stage
table, telescoping sum, and N-turn aggregation. No I/O -- fully unit-tested."""
from __future__ import annotations

import numpy as np

# (label, anchor_key, default source tag, .env lever that moves this stage)
# Anchors are t0-relative seconds. 'capture' is handled separately as a pre-t0 offset.
_WATERFALL_STAGES = [
    ("STT finalize -> LLM",                  "llm_recv",       "log",           "STT provider"),
    ("LLM first token",                      "llm_ttfb",       "log",           "OPENROUTER_PROVIDER_ONLY / model"),
    ("LLM -> TTS (sentence-1 flush)",        "tts_recv",       "log",           "COSYVOICE_FIRST_PIECE*"),
    ("TTS synth first chunk",                "tts_ttfb",       "log",           "first-piece / CUDA graphs / model / hop"),
    ("Avatar render (first frame)",          "render",         "log",           "MUSETALK_TRT / MUSETALK_BATCH"),
    ("TTS -> bot-start (steady lead-hold)",  "bot_started",    "log",           "MUSETALK_LEAD_FRAMES / MUSETALK_FEED_BURST_S"),
    ("Transport + encode + network",         "client_arrival", "probe",         "WEBRTC_VIDEO_BITRATE_MAX / network"),
    ("Browser jitter buffer",                "jitter",         "browser-stats", "CLIENT_JITTER_BUFFER_MS"),
    ("Browser decode + playout",             "playout",        "browser-audio", "device / OS"),
]
# The keys the harness measures per turn (used by aggregate_turns + __main__).
ANCHOR_KEYS = [s[1] for s in _WATERFALL_STAGES]


def answer_onset_epoch(samples, t0_epoch, guard=0.15, thresh_frac=0.18, run=3):
    """First SUSTAINED energetic audio frame after t0 = the answer reaching the client.
    samples: list of (arrival_epoch, rms). Returns the onset epoch or None."""
    win = [(t, r) for (t, r) in samples if t >= t0_epoch + guard]
    if len(win) < run:
        return None
    peak = max(r for _, r in win)
    if peak <= 0:
        return None
    thr = thresh_frac * peak
    for i in range(len(win) - run + 1):
        if all(win[i + k][1] >= thr for k in range(run)):
            return win[i][0]
    return None


def build_waterfall(anchors, playout_source="est", capture=None):
    """Per-stage latency from the true mic moment to the user's ear.
    anchors: dict of t0-relative offsets (s); a None or NEGATIVE anchor -> 'unknown'
    row that does NOT corrupt the running sum (the next known stage's delta absorbs
    the gap, so ok-row deltas telescope to the last known cum).
    capture: t0 - speech_end (s), the pre-t0 VAD-hangover+turn-end cost, or None."""
    rows = []
    # Pre-t0 capture row. offset = how much to add to every downstream cum.
    if capture is not None and capture >= 0:
        offset = capture
        rows.append(dict(stage="Capture: speech-end -> t0 (VAD hangover + turn-end)",
                         delta=round(capture, 3), cum=round(capture, 3),
                         source="driver", lever="VAD_STOP_SECS / turn strategy", status="ok"))
        prev = capture
    else:
        offset = 0.0
        rows.append(dict(stage="Capture: speech-end -> t0 (VAD hangover + turn-end)",
                         delta=None, cum=None, source="driver",
                         lever="VAD_STOP_SECS / turn strategy", status="unknown"))
        prev = 0.0

    for label, key, source, lever in _WATERFALL_STAGES:
        if key == "playout":
            source = playout_source
        end = anchors.get(key)
        if end is None or end < 0:
            rows.append(dict(stage=label, delta=None, cum=None, source=source,
                             lever=lever, status="unknown"))
            continue
        cum = end + offset
        rows.append(dict(stage=label, delta=round(cum - prev, 3), cum=round(cum, 3),
                         source=source, lever=lever, status="ok"))
        prev = cum
    total = next((r["cum"] for r in reversed(rows) if r["cum"] is not None), None)
    rows.append(dict(stage="END-TO-END, user hears", delta=None, cum=total,
                     source="", lever="", status="total"))
    return rows


def _median(xs):
    xs = sorted(v for v in xs if v is not None)
    if not xs:
        return None
    return round(float(np.median(xs)), 3)


def _p95(xs):
    xs = sorted(v for v in xs if v is not None)
    if not xs:
        return None
    return round(float(np.percentile(xs, 95)), 3)


def aggregate_turns(turns, keys):
    """Per-key median + p95 across turns, plus a fresh (turns 0-1) vs warm (turns >=4,
    or last 2 if fewer than 5) split so the session-degradation shows up explicitly."""
    def sub(subset):
        return {k: _median([t.get(k) for t in subset]) for k in keys}
    warm_idx = turns[4:] if len(turns) >= 5 else turns[-2:]
    return {
        "median": {k: _median([t.get(k) for t in turns]) for k in keys},
        "p95": {k: _p95([t.get(k) for t in turns]) for k in keys},
        "fresh": sub(turns[:2]),
        "warm": sub(warm_idx),
    }
```

- [ ] **Step 4: Create `scripts/measure/__init__.py`**

```python
"""Mic-to-ear latency harness. Public entry: python -m scripts.measure.
Pure functions re-exported here so callers/tests import from scripts.measure."""
from scripts.measure.waterfall import (
    ANCHOR_KEYS,
    aggregate_turns,
    answer_onset_epoch,
    build_waterfall,
)
from scripts.measure.logparse import parse_playout_beacon

__all__ = ["ANCHOR_KEYS", "aggregate_turns", "answer_onset_epoch",
           "build_waterfall", "parse_playout_beacon"]
```

Note: this imports `logparse` — Task 2 creates it. Until then, temporarily comment the `logparse` import line to run Task 1's tests, then restore in Task 2. (The archive test needs `parse_playout_beacon` only after Task 2.)

- [ ] **Step 5: Run the new tests (with the logparse import temporarily commented)**

Run: `pytest archive/_measure_waterfall_test.py -k "capture or new_stages or aggregate" -v`
Expected: PASS (5 new cases).

- [ ] **Step 6: Commit**

```bash
git add scripts/measure/__init__.py scripts/measure/waterfall.py archive/_measure_waterfall_test.py
git commit -m "measure: waterfall stage table + capture offset + N-turn aggregation"
```

---

### Task 2: `logparse.py` — per-turn anchors + beacon parsing

**Files:**
- Create: `scripts/measure/logparse.py`
- Test: `archive/_measure_logparse_test.py`
- Modify: `scripts/measure/__init__.py` (restore the `logparse` import)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces:
  - `parse_lines(path) -> list[tuple[datetime, str]]`.
  - `find_turn_indices(lines) -> list[int]` (indices of `[TTFO` lines).
  - `build_turn(lines, bi) -> dict` — anchors for the turn whose bot-start is at `lines[bi]`: keys `t0` (datetime), `ttfo_s`, `ttfo_pass`, `bot_started`, `question`, `user_started`, `llm_recv`, `llm_ttfb`, `tts_recv`, `tts_ttfb`, `render`, `bot_stopped`, `sentences`, plus `t0_epoch`.
  - `parse_playout_beacon(lines, t0) -> float | None` — onset offset (s) from the latest `[client-playout]` line (accepts `{"ev":"audio-onset","t":<ms>}` OR `{"onset":<ms>}`).
  - `parse_beacon_full(lines, t0) -> dict | None` — `{onset, recv, jitter, rtt}` all t0-relative seconds (jitter/rtt in seconds), from the latest `[client-playout]` line whose fields are present.

- [ ] **Step 1: Write failing tests**

Create `archive/_measure_logparse_test.py`:

```python
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
        ("12:00:02.750", "[TTFO OK] 2.75s (target 3.0s)"),
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
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest archive/_measure_logparse_test.py -v`
Expected: import error (module not present).

- [ ] **Step 3: Implement `scripts/measure/logparse.py`**

```python
"""pipeline.log -> per-turn latency anchors, on the pipeline's single wall clock.
All numeric anchors are t0-relative seconds (t0 = user stopped speaking)."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
LOG = ROOT / "logs" / "pipeline.log"

_TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) \| ")


def parse_lines(path=LOG):
    out = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            mt = _TS.match(ln)
            if mt:
                dt = datetime.strptime(mt.group(1), "%Y-%m-%d %H:%M:%S.%f")
                out.append((dt, ln.rstrip("\n")))
    return out


def find_turn_indices(lines):
    return [i for i, (_, t) in enumerate(lines) if "[TTFO" in t]


def build_turn(lines, bi, target_s=3.0):
    """Extract anchors for the turn whose [TTFO] line is at index bi."""
    bot_started_t = lines[bi][0]
    ttfo_s = ttfo_pass = None
    mt = re.search(r"\[TTFO (OK |OVER)\] ([\d.]+)s", lines[bi][1])
    if mt:
        ttfo_pass = mt.group(1).strip() == "OK"
        ttfo_s = float(mt.group(2))

    # t0 = the 'Generating chat' / 'User stopped speaking' just before the bot start.
    t0 = question = None
    for dt, txt in lines[:bi][::-1]:
        if t0 is None and ("Generating chat from context" in txt or "User stopped speaking" in txt):
            t0 = dt
        if question is None:
            qm = re.search(r"'role': 'user', 'content': '(.*?)'\}\]", txt)
            if qm:
                question = qm.group(1)
        if t0 and question:
            break
    if t0 is None:
        t0 = bot_started_t

    def off(dt):
        return round((dt - t0).total_seconds(), 3)

    if ttfo_s is None:
        ttfo_s = round((bot_started_t - t0).total_seconds(), 2)
        ttfo_pass = ttfo_s <= target_s

    turn = dict(t0=t0, t0_epoch=t0.timestamp(), ttfo_s=ttfo_s, ttfo_pass=ttfo_pass,
                bot_started=off(bot_started_t), question=question)

    win = [(dt, txt) for dt, txt in lines if -3 <= (dt - t0).total_seconds() <= 60]
    user_started = llm_recv = llm_ttfb = render = bot_stopped = None
    sentences, tts_ttfb = [], []
    for dt, txt in win:
        if "User started speaking" in txt and user_started is None:
            user_started = off(dt)
        if llm_recv is None and "Generating chat from context" in txt and dt >= t0:
            llm_recv = off(dt)
        if "OpenAILLMService" in txt and "TTFB:" in txt and llm_ttfb is None:
            llm_ttfb = (off(dt), float(re.search(r"TTFB: ([\d.]+)s", txt).group(1)))
        m1 = re.search(r"run_tts:\d+ - CosyVoice TTS \[(.*)\]", txt)
        if m1 and dt >= t0:
            sentences.append((off(dt), m1.group(1)))
        if "CosyVoiceTTSService" in txt and "TTFB:" in txt and dt >= t0:
            tts_ttfb.append((off(dt), float(re.search(r"TTFB: ([\d.]+)s", txt).group(1))))
        if render is None and "[render] first-frame" in txt and dt >= t0:
            render = off(dt)
        if "Bot stopped speaking based on TTSStoppedFrame" in txt:
            bot_stopped = off(dt)

    turn.update(user_started=user_started,
                llm_recv=llm_recv if llm_recv is not None else 0.0,
                llm_ttfb=llm_ttfb, render=render, bot_stopped=bot_stopped,
                tts_recv=sentences[0][0] if sentences else None,
                tts_ttfb=tts_ttfb, sentences=sentences)
    return turn


def _latest_beacon(lines, t0):
    """Return the parsed json of the last [client-playout] line at/after t0-1s, or None."""
    for dt, txt in reversed(lines):
        if "[client-playout]" not in txt:
            continue
        if (dt - t0).total_seconds() < -1.0:
            break
        m = re.search(r"\[client-playout\]\s*(\{.*\})", txt)
        if m:
            try:
                return json.loads(m.group(1))
            except ValueError:
                continue
    return None


def parse_playout_beacon(lines, t0):
    """Onset offset (s) from the latest beacon; accepts old {'t':ms} or new {'onset':ms}."""
    b = _latest_beacon(lines, t0)
    if not b:
        return None
    ms = b.get("onset", b.get("t"))
    return None if ms is None else round(ms / 1000.0 - t0.timestamp(), 6)


def parse_beacon_full(lines, t0):
    """All beacon fields, t0-relative seconds. jitter/rtt are durations (ms -> s)."""
    b = _latest_beacon(lines, t0)
    if not b:
        return None
    base = t0.timestamp()
    def rel(ms):
        return None if ms is None else round(ms / 1000.0 - base, 6)
    def dur(ms):
        return None if ms is None else round(ms / 1000.0, 6)
    return dict(recv=rel(b.get("recv")), onset=rel(b.get("onset", b.get("t"))),
                jitter=dur(b.get("jitterMs")), rtt=dur(b.get("rttMs")))
```

- [ ] **Step 4: Restore the logparse import in `scripts/measure/__init__.py`** (uncomment the line from Task 1 Step 4).

- [ ] **Step 5: Run all measure tests**

Run: `pytest archive/_measure_logparse_test.py archive/_measure_waterfall_test.py -v`
Expected: PASS (all logparse cases + all 12 waterfall cases, including the previously-red `parse_playout_beacon` case).

- [ ] **Step 6: Commit**

```bash
git add scripts/measure/logparse.py scripts/measure/__init__.py archive/_measure_logparse_test.py
git commit -m "measure: per-turn log anchors (+STT-finalize +render) and beacon parse"
```

---

### Task 3: `[render]` first-frame log line in the avatar client

**Files:**
- Modify: `local_services/musetalk_video.py` (the `_on_frame` / video-marker handler that already computes `[avatar offset]`).

**Interfaces:**
- Produces: a single `logger.info(f"[render] first-frame +{ms:.0f}ms")` per turn = ms from the turn's first outbound voice chunk to the first REAL rendered frame (`video_start`/first `kind==0` frame). Consumed by `logparse.build_turn` (`render` anchor).

- [ ] **Step 1: Locate the turn-start + first-real-frame points**

Run: `grep -n "video_start\|_on_frame\|speech_start\|def run_tts\|first" local_services/musetalk_video.py | head -40`
Read the handler that receives `video_start` and the point where the turn's first PCM chunk is sent. Identify a per-turn `self._turn_send_t0` (set when the first audio chunk of a turn is forwarded) and the first real-frame receipt.

- [ ] **Step 2: Add the instrumentation (the actual edit)**

Add a per-turn state field in `__init__`: `self._render_logged = False` and `self._turn_audio_t0 = None`.
Where the turn's first audio chunk is forwarded to the server (turn start), set:
```python
if self._turn_audio_t0 is None:
    self._turn_audio_t0 = time.monotonic()
    self._render_logged = False
```
Where the first REAL rendered frame of the turn is received (the `video_start` marker or first `kind==0` frame), add:
```python
if not self._render_logged and self._turn_audio_t0 is not None:
    self._render_logged = True
    ms = (time.monotonic() - self._turn_audio_t0) * 1000.0
    logger.info(f"[render] first-frame +{ms:.0f}ms")
```
Reset `self._turn_audio_t0 = None` at turn end / on `reset` (reuse the existing per-turn reset path so a new turn re-arms). Keep it ASCII, f-string (loguru brace-style).

- [ ] **Step 3: Verify it logs (manual smoke, needs the stack up)**

Run the avatar server + pipeline, drive one turn:
`python -m scripts._webrtc_probe --mic output/_zh_q_def.wav --lead 8 --duration 30`
Then: `grep -n "\[render\] first-frame" logs/pipeline.log | tail -3`
Expected: one `[render] first-frame +NNNms` line per turn, NNN in a sane range (tens-to-hundreds ms).

- [ ] **Step 4: Commit**

```bash
git add local_services/musetalk_video.py
git commit -m "avatar: log [render] first-frame per turn for the latency waterfall"
```

---

### Task 4: beacon endpoint in the pipeline

**Files:**
- Modify: `pipeline/main.py` (the `_inject_client_patches` middleware near the `/client/say` and `/client/ice-config` handlers, ~lines 538-640).

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `POST /client/measure-beacon` — body `{"onset":ms,"recv":ms,"jitterMs":n,"rttMs":n}`; the handler `logger.info(f"[client-playout] {json.dumps(body)}")` and returns `{"ok": true}`. (Landing it in `pipeline.log` is how `logparse` reads it on the pipeline clock.)
  - `GET /client/measure-beacon` returns `{"serverEpochMs": <now ms>}` for the client's NTP-style min-RTT clock-offset handshake.
  - `GET /client/ice-config` payload gains `"measureBeacon": <bool from MEASURE_BEACON env, default true>` and `"serverEpochMs": <now ms>`.

- [ ] **Step 1: Add the handlers in the middleware**

Inside `_inject_client_patches`, mirroring the `/client/say` block, add (ASCII, async, non-blocking):
```python
if request.url.path == "/client/measure-beacon":
    if request.method == "POST":
        try:
            body = await request.json()
        except Exception:
            body = {}
        # Land it in pipeline.log so scripts.measure reads it on the pipeline clock.
        logger.info(f"[client-playout] {json.dumps(body)}")
        return JSONResponse({"ok": True})
    return JSONResponse({"serverEpochMs": time.time() * 1000.0})
```
Ensure `json` and `time` are imported at module top (they are; verify). In the `GET /client/ice-config` payload dict, add:
```python
"measureBeacon": os.getenv("MEASURE_BEACON", "1") != "0",
"serverEpochMs": time.time() * 1000.0,
```

- [ ] **Step 2: Test the endpoint (unit, no full stack)**

Create `archive/_measure_beacon_test.py`:
```python
"""POST a beacon, assert it lands in the log as [client-playout]; GET returns serverEpochMs.
Run: pytest archive/_measure_beacon_test.py -v  (starts nothing; uses Starlette TestClient)."""
import json
from scripts.measure.logparse import parse_beacon_full
from datetime import datetime


def test_beacon_json_roundtrips_through_logparse():
    # The endpoint logs exactly this line shape; logparse must read it back.
    body = {"onset": 1751800000200, "recv": 1751800000050, "jitterMs": 90, "rttMs": 20}
    line = f"2026-07-16 12:00:00.500 | INFO | [client-playout] {json.dumps(body)}"
    t0 = datetime.fromtimestamp(1751800000.0)
    b = parse_beacon_full([(t0, line)], t0)
    assert abs(b["onset"] - 0.200) < 1e-6 and abs(b["jitter"] - 0.090) < 1e-6
```
Run: `pytest archive/_measure_beacon_test.py -v` -> PASS. (This locks the log-line contract; the live HTTP path is smoke-tested in Task 10.)

- [ ] **Step 3: Preflight the import graph**

Run: `python -m scripts.preflight`
Expected: no import errors (pipecat drift check for the edited `main.py`).

- [ ] **Step 4: Commit**

```bash
git add pipeline/main.py archive/_measure_beacon_test.py
git commit -m "pipeline: /client/measure-beacon endpoint + ice-config clock handshake"
```

---

### Task 5: `?measure=1` beacon in the `/studio` client

**Files:**
- Modify: `local_services/studio_client/index.html` (near the `AudioContext`/`createMediaStreamSource` at ~866-885 and `pc.ontrack` at ~930).

**Interfaces:**
- Consumes: `GET /client/ice-config` (`measureBeacon`, `serverEpochMs`), `GET/POST /client/measure-beacon`.
- Produces: on turn end, `POST /client/measure-beacon {onset, recv, jitterMs, rttMs}` when `?measure=1` (or `measureBeacon` config) is set.

- [ ] **Step 1: Add the beacon JS (guarded so normal sessions pay nothing)**

Add near the top-level script state:
```javascript
const MEASURE = new URLSearchParams(location.search).get('measure') === '1';
let clockOffMs = 0;     // browser Date.now() - server epoch (min-RTT estimate)
async function syncClock(){
  let best = 1e9;
  for (let i=0;i<5;i++){
    const t0 = Date.now();
    const r = await fetch('/client/measure-beacon'); const j = await r.json();
    const t1 = Date.now(); const rtt = t1 - t0;
    if (rtt < best){ best = rtt; clockOffMs = (t0 + rtt/2) - j.serverEpochMs; }
  }
}
```
In `pc.ontrack`, when the audio stream arrives and `MEASURE`, hang an analyser on the SAME audioCtx already built for the speaker route, and record first-sound onset + rolling getStats:
```javascript
if (MEASURE && track.kind === 'audio'){
  audioCtx = audioCtx || new (window.AudioContext||window.webkitAudioContext)();
  const an = audioCtx.createAnalyser(); an.fftSize = 512;
  audioCtx.createMediaStreamSource(stream).connect(an);
  const buf = new Uint8Array(an.fftSize);
  let recvMs = null, onsetMs = null, peak = 0;
  const tick = () => {
    an.getByteTimeDomainData(buf);
    let s = 0; for (const v of buf){ const d = v-128; s += d*d; }
    const rms = Math.sqrt(s/buf.length);
    const now = Date.now() - clockOffMs;
    if (recvMs === null && rms > 1) recvMs = now;
    peak = Math.max(peak, rms);
    if (onsetMs === null && peak > 0 && rms >= 0.18*peak && rms > 3) onsetMs = now;
    if (onsetMs === null) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
  // On turn end (bot stops), read jitter/rtt from getStats and POST once.
  window._postBeacon = async () => {
    let jitterMs = 0, emit = 0, rttMs = 0;
    const stats = await pc.getStats();
    stats.forEach(r => {
      if (r.type === 'inbound-rtp' && r.kind === 'audio'){
        if (r.jitterBufferDelay != null && r.jitterBufferEmittedCount)
          jitterMs = 1000 * r.jitterBufferDelay / r.jitterBufferEmittedCount;
      }
      if (r.type === 'candidate-pair' && r.currentRoundTripTime != null)
        rttMs = 1000 * r.currentRoundTripTime;
    });
    if (recvMs !== null)
      fetch('/client/measure-beacon', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({recv: recvMs, onset: onsetMs, jitterMs: Math.round(jitterMs), rttMs: Math.round(rttMs)})});
  };
}
```
Call `syncClock()` once after connect when `MEASURE`. Fire `window._postBeacon()` when the turn's audio settles — reuse the existing "bot stopped" / transcript-commit hook already in the client; if none is convenient, POST 1.5s after `onsetMs` is first set. Keep the getStats reset per turn (recvMs/onsetMs/peak nulled at each new turn start).

- [ ] **Step 2: Manual smoke (static files -> just reload)**

Open `http://localhost:7860/studio/?measure=1`, speak/type one turn, then:
`grep -n "\[client-playout\]" logs/pipeline.log | tail -2`
Expected: one `[client-playout] {"recv":...,"onset":...,"jitterMs":...,"rttMs":...}` line with plausible values (jitterMs tens-to-~150).

- [ ] **Step 3: Commit**

```bash
git add local_services/studio_client/index.html
git commit -m "studio: ?measure=1 playout beacon (WebAudio onset + getStats jitter/rtt)"
```

---

### Task 6: `drive.py` — headless probe + Playwright browser drivers

**Files:**
- Create: `scripts/measure/drive.py`

**Interfaces:**
- Consumes: `scripts._webrtc_probe.build_mic_wav`, `wait_ice`, `lip_offset_from_mp4`.
- Produces:
  - `async run_probe(mic_wav, lead, tail, duration) -> (vwall, awall, connect_t)` (moved verbatim from current `measure.py`).
  - `probe_metrics(vwall, awall, connect_t, fps) -> dict` (moved verbatim).
  - `async run_browser_turns(mic_wav, n_turns, lead, gap, question_wavs=None) -> bool` — launches headless Chromium via Playwright with the wav as fake mic, opens `/studio/?measure=1`, drives `n_turns` turns, waits for each to settle. Returns True if it ran, False if Playwright/Chromium unavailable (caller falls back to `run_probe`).
  - `speech_end_epoch(mic_wav, lead, play_start_epoch) -> float` — `play_start_epoch + lead + speech_duration(mic_wav)`; the capture-segment anchor.

- [ ] **Step 1: Move `run_probe` + `probe_metrics` verbatim** from `scripts/measure.py` into `drive.py` (imports: `aiohttp, asyncio, time, numpy as np`, aiortc pieces, and `from scripts._webrtc_probe import build_mic_wav, lip_offset_from_mp4, wait_ice`). Keep `OFFER_URL`, `MP4`, `_audio_rms` local to `drive.py`.

- [ ] **Step 2: Add `speech_end_epoch`**

```python
import wave

def speech_duration(mic_wav):
    with wave.open(mic_wav, "rb") as w:
        return w.getnframes() / float(w.getframerate())

def speech_end_epoch(mic_wav, lead, play_start_epoch):
    """When the user's audio goes silent, in epoch seconds: the probe/browser starts
    playing at play_start_epoch, the wav is [lead silence | speech | tail silence]."""
    return play_start_epoch + lead + speech_duration(mic_wav)
```

- [ ] **Step 3: Add the Playwright driver**

```python
async def run_browser_turns(mic_wav, n_turns, lead=1.0, gap=6.0):
    """Real headless Chromium: wav as fake mic, /studio/?measure=1, drive n_turns.
    Returns False (caller falls back to the aiortc probe) if Playwright is unavailable."""
    try:
        from playwright.async_api import async_playwright
    except Exception:
        print("  [browser] Playwright not installed -- falling back to headless probe.")
        return False
    # Chrome needs a 16-bit PCM wav; build_mic_wav already emits one. Fake-mic flag wants a path.
    driven = build_mic_wav(mic_wav, lead, tail=2.0)
    args = ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
            "--autoplay-policy=no-user-gesture-required",
            f"--use-file-for-fake-audio-capture={driven}"]
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=args)
            page = await browser.new_page()
            await page.goto("http://localhost:7860/studio/?measure=1")
            await page.click("text=/connect/i", timeout=8000)  # the mic/connect button
            for _ in range(n_turns):
                # A typed turn is deterministic and re-drivable; the beacon still measures playout.
                await page.fill("input[type=text], textarea", "Please explain this briefly.")
                await page.keyboard.press("Enter")
                await page.wait_for_timeout(int(gap * 1000))
            await browser.close()
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [browser] driver error ({e!r}); falling back to headless probe.")
        return False
```
Note: the exact selectors (`connect` button, text input) must match `/studio` — verify against `studio_client/index.html` and adjust the selectors in this step (they are the only page-specific bits).

- [ ] **Step 4: Import smoke**

Run: `python -c "import scripts.measure.drive as d; print(d.speech_duration('output/_zh_q_def.wav'))"`
Expected: prints the wav duration (a float), no import error.

- [ ] **Step 5: Commit**

```bash
git add scripts/measure/drive.py
git commit -m "measure: probe + Playwright browser drivers, speech-end anchor"
```

---

### Task 7: `report.py` — writers, history, compare, summary

**Files:**
- Create: `scripts/measure/report.py`

**Interfaces:**
- Consumes: `waterfall._WATERFALL_STAGES`, a turn dict, aggregated anchors.
- Produces:
  - `build_events(turn) -> list[dict]`, `build_handoffs(turn) -> list[dict]`, `build_metrics(turn, pm, offline_lip) -> list[dict]` (moved from current `measure.py`; `build_events` gains a capture bar + browser-tail events).
  - `write_outputs(report) -> None` (writes `output/measure_report.json` + `docs/measure_data.js`, verbatim from current `measure.py`).
  - `append_history(report, env_knobs) -> None` — appends one line to `output/measure_history.jsonl`.
  - `read_history(n=30) -> list[dict]`.
  - `compare_runs(a_idx, b_idx) -> None` — prints a per-stage delta table between two history rows.
  - `print_summary(report) -> None` (extended: prints the lever column + fresh/warm split).
  - `ENV_KNOBS: list[str]` — the latency-relevant env names to snapshot.

- [ ] **Step 1: Move `build_events`, `build_handoffs`, `build_metrics`, `write_outputs`, `print_summary`** from `scripts/measure.py`. Update `print_summary`'s waterfall loop to also print `r["lever"]`. Update the `JSON_OUT`/`JS_OUT`/`ROOT` constants to resolve from `report.py`'s location (`ROOT = Path(__file__).resolve().parent.parent.parent`).

- [ ] **Step 2: Add history + compare**

```python
import json, subprocess, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
HISTORY = ROOT / "output" / "measure_history.jsonl"
ENV_KNOBS = ["LANGUAGE", "COSYVOICE_MODEL", "OPENROUTER_PROVIDER_ONLY", "OPENROUTER_MODEL",
             "MUSETALK_LEAD_FRAMES", "MUSETALK_FPS", "MUSETALK_TRT", "COSYVOICE_FIRST_PIECE",
             "COSYVOICE_FIRST_PIECE_ZH", "FILLER_WORDS", "VAD_STOP_SECS", "CLIENT_JITTER_BUFFER_MS"]

def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"

def append_history(report, env_knobs):
    row = dict(when=report["meta"]["when"], commit=_git_commit(), env=env_knobs,
               e2e_median=report["meta"].get("e2e_median"),
               stage_medians=report["meta"].get("stage_medians", {}))
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def read_history(n=30):
    if not HISTORY.exists():
        return []
    rows = [json.loads(l) for l in HISTORY.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[-n:]

def compare_runs(a_idx, b_idx):
    rows = read_history(9999)
    a, b = rows[a_idx], rows[b_idx]
    print(f"compare  A={a['when']} ({a['commit']})   B={b['when']} ({b['commit']})")
    keys = sorted(set(a.get("stage_medians", {})) | set(b.get("stage_medians", {})))
    for k in keys:
        av, bv = a["stage_medians"].get(k), b["stage_medians"].get(k)
        if av is None or bv is None:
            print(f"  {k:<40} A={av}  B={bv}")
        else:
            print(f"  {k:<40} {av:+.3f} -> {bv:+.3f}  ({bv-av:+.3f}s)")
```

- [ ] **Step 3: Unit test history round-trip**

Add to `archive/_measure_waterfall_test.py`:
```python
def test_history_append_and_read(tmp_path, monkeypatch):
    import scripts.measure.report as rep
    monkeypatch.setattr(rep, "HISTORY", tmp_path / "h.jsonl")
    rep.append_history({"meta": {"when": "2026-07-16 12:00", "e2e_median": 2.9,
                                 "stage_medians": {"LLM first token": 0.7}}}, {"LANGUAGE": "zh"})
    rows = rep.read_history()
    assert rows[-1]["e2e_median"] == 2.9 and rows[-1]["env"]["LANGUAGE"] == "zh"
```
Run: `pytest archive/_measure_waterfall_test.py -k history -v` -> PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/measure/report.py archive/_measure_waterfall_test.py
git commit -m "measure: report writers + run-history + compare"
```

---

### Task 8: `__main__.py` — orchestration; delete `scripts/measure.py`

**Files:**
- Create: `scripts/measure/__main__.py`
- Delete: `scripts/measure.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `python -m scripts.measure [--turns N] [--browser/--no-browser] [--mic ...] [--compare A B] ...` writing `output/measure_report.json`, `docs/measure_data.js`, appending `output/measure_history.jsonl`.

- [ ] **Step 1: Write `__main__.py`** — orchestrate: for each of N turns, drive (browser if available else probe), record `play_start_epoch`; after all turns, `parse_lines()` once, `find_turn_indices()`, take the last N turns, `build_turn` each, compute per-turn `capture = t0_epoch - speech_end_epoch`, attach `parse_beacon_full` (browser F rows) or `answer_onset_epoch` (probe E row) per turn, `aggregate_turns(...)` over `ANCHOR_KEYS + ["capture"]`, `build_waterfall(median_anchors, playout_source, capture=median_capture)`, assemble `report` (meta gets `e2e_median`, `stage_medians`, `fresh`/`warm`), `write_outputs`, `append_history`, `print_summary`. Handle `--compare A B` as an early branch calling `report.compare_runs`. Keep the CLI args from the old `measure.py` (`--mic --lead --tail --duration --fps --offline-capture ...`) plus `--turns` (default 5) and `--browser/--no-browser` (default: try browser).

Provide the full file (the engineer copies it):
```python
"""Unified mic-to-ear latency harness. Drives N turns, parses the pipeline.log delta,
measures real browser playout (Playwright or /studio beacon), and writes one report +
docs/measure_data.js + appends output/measure_history.jsonl.

Run (stack up, no browser tab on /studio):
    python -m scripts.measure --turns 5
    python -m scripts.measure --no-browser --offline-capture
    python -m scripts.measure --compare -2 -1
"""
from __future__ import annotations
import argparse, asyncio, os, sys, time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from scripts.measure import ANCHOR_KEYS, aggregate_turns, answer_onset_epoch, build_waterfall
from scripts.measure import drive, logparse, report

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")


async def _drive(args):
    """Drive N turns; return list of (play_start_epoch,) markers. Prefer the real browser."""
    starts = []
    if args.browser:
        t0 = time.time()
        ok = await drive.run_browser_turns(args.mic, args.turns, lead=1.0, gap=args.gap)
        if ok:
            # browser drove all turns back-to-back from t0; approximate each start.
            return [t0 + i * (args.gap) for i in range(args.turns)], "browser"
    # Fallback: headless probe, one turn per capture window.
    for _ in range(args.turns):
        ps = time.time()
        await drive.run_probe(args.mic, args.lead, args.tail, args.duration)
        starts.append(ps)
    return starts, "probe"


def _turn_anchor_dict(turn, capture, beacon, onset_rel):
    d = {k: (turn.get(k)[0] if isinstance(turn.get(k), tuple) else turn.get(k))
         for k in ANCHOR_KEYS}
    # tts_ttfb/llm_ttfb are stored as tuples/lists -> take the offset.
    d["llm_ttfb"] = turn["llm_ttfb"][0] if turn.get("llm_ttfb") else None
    d["tts_ttfb"] = turn["tts_ttfb"][0][0] if turn.get("tts_ttfb") else None
    d["client_arrival"] = (beacon.get("recv") if beacon else onset_rel)
    d["jitter"] = (d["client_arrival"] + beacon["jitter"]) if (beacon and beacon.get("jitter") is not None and d["client_arrival"] is not None) else None
    d["playout"] = (beacon.get("onset") if beacon else None)
    d["capture"] = capture
    return d


def main():
    ap = argparse.ArgumentParser(description="Mic-to-ear latency harness.")
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--browser", dest="browser", action="store_true", default=True)
    ap.add_argument("--no-browser", dest="browser", action="store_false")
    ap.add_argument("--mic", default="output/_zh_q_def.wav")
    ap.add_argument("--lead", type=float, default=8.0)
    ap.add_argument("--tail", type=float, default=28.0)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--gap", type=float, default=8.0)
    ap.add_argument("--fps", type=int, default=14)
    ap.add_argument("--offline-capture", action="store_true")
    ap.add_argument("--compare", nargs=2, type=int, metavar=("A", "B"))
    ap.add_argument("--machine", default="this box (RTX 5060 Ti, Blackwell)")
    ap.add_argument("--stack", default="Deepgram STT - OpenRouter LLM - CosyVoice (vLLM/WSL) TTS - MuseTalk avatar")
    args = ap.parse_args()

    if args.compare:
        report.compare_runs(*args.compare); return

    print(f"[1/3] driving {args.turns} turns through the live pipeline...")
    starts, path_kind = asyncio.run(_drive(args))

    print("[2/3] parsing pipeline.log for the driven turns...")
    lines = logparse.parse_lines()
    idxs = logparse.find_turn_indices(lines)
    turns_raw = [logparse.build_turn(lines, i) for i in idxs[-args.turns:]]

    anchor_dicts = []
    for turn, ps in zip(turns_raw, starts[-len(turns_raw):]):
        se = drive.speech_end_epoch(args.mic, args.lead, ps)
        capture = round(turn["t0_epoch"] - se, 3)
        capture = capture if capture >= 0 else None
        beacon = logparse.parse_beacon_full(lines, turn["t0"])
        onset_rel = None  # probe arrival is computed from the last-turn awall if needed
        anchor_dicts.append(_turn_anchor_dict(turn, capture, beacon, onset_rel))

    keys = ANCHOR_KEYS + ["capture"]
    agg = aggregate_turns(anchor_dicts, keys)
    med = agg["median"]
    rows = build_waterfall(med, playout_source=("browser-audio" if path_kind == "browser" else "est"),
                           capture=med.get("capture"))
    last = turns_raw[-1]
    stage_medians = {r["stage"]: r["cum"] for r in rows if r["status"] == "ok"}
    e2e = next((r["cum"] for r in reversed(rows) if r["status"] == "total"), None)

    rep = {
        "meta": {"when": last["t0"].strftime("%Y-%m-%d %H:%M"), "question": last["question"],
                 "machine": args.machine, "stack": args.stack,
                 "ttfo": last["ttfo_s"], "ttfo_target": 3.0, "ttfo_pass": last["ttfo_pass"],
                 "turns": len(turns_raw), "e2e_median": e2e, "stage_medians": stage_medians,
                 "fresh": agg["fresh"], "warm": agg["warm"], "p95": agg["p95"]},
        "events": report.build_events(last),
        "handoffs": report.build_handoffs(last),
        "metrics": report.build_metrics(last, {}, None),
        "waterfall": rows,
        "raw": {"anchors_per_turn": anchor_dicts, "agg": agg},
    }
    report.write_outputs(rep)
    report.append_history(rep, {k: os.getenv(k) for k in report.ENV_KNOBS})
    report.print_summary(rep)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
```

- [ ] **Step 2: Delete the old monolith**

```bash
git rm scripts/measure.py
```

- [ ] **Step 3: Full test suite + preflight**

Run: `pytest archive/_measure_waterfall_test.py archive/_measure_logparse_test.py archive/_measure_beacon_test.py -v && python -m scripts.preflight`
Expected: all green; preflight OK.

- [ ] **Step 4: Import smoke**

Run: `python -c "import scripts.measure.__main__"` and `python -m scripts.measure --compare 0 0 2>&1 | head` (compare will error only if history empty -- acceptable; the import must be clean).

- [ ] **Step 5: Commit**

```bash
git add scripts/measure/__main__.py
git rm scripts/measure.py
git commit -m "measure: orchestration entrypoint (N turns, browser playout, history); drop monolith"
```

---

### Task 9: HTML — lever column, source tags, capture + browser tail, sparkline

**Files:**
- Modify: `docs/workflow-timeline.html`

**Interfaces:**
- Consumes: `docs/measure_data.js` `window.MEASURE` (now with `waterfall[].lever`, `waterfall[].source`, `meta.fresh/warm/p95/e2e_median`, and history via a new `meta.history` array the report may include).

- [ ] **Step 1: Waterfall cards -> show lever + source tag.** In `buildWaterfall()`, add the lever line and a source chip:
```javascript
const lever = r.lever ? `<div class="n" style="color:var(--mut)">lever: ${r.lever}</div>` : '';
const srcTag = r.source ? `<span class="stag" style="color:var(--dim)">${r.source}</span>` : '';
card.innerHTML = `<div class="k">${r.stage} ${srcTag}</div>`+
  `<div class="v${tag}">${val}</div>`+`<div class="n">${sub}</div>`+lever;
```

- [ ] **Step 2: Fresh-vs-warm degradation callout.** After the waterfall grid, inject a small block if `meta.warm` and `meta.fresh` exist, showing the e2e/LLM/TTS deltas warm-minus-fresh (the session-degradation number).

- [ ] **Step 3: Run-history sparkline.** If `window.MEASURE.meta.history` is present (array of `{when, e2e_median}`), draw a tiny inline SVG polyline near the banner. Keep it dependency-free (hand-built SVG, same as the rest of the page).

- [ ] **Step 4: Swimlane + copy.** Add the `capture` lane bar (pre-t0, from `meta` capture) and browser-tail markers to `build_events` output (already produced in `report.py`); update the two stale `<8s`/`target 8s` references to `<3s`/`target 3s`, and the footer "live/audio-master" note to "steady/video-master (default)".

- [ ] **Step 5: Verify render.** Open `docs/workflow-timeline.html` in a browser after a real run; confirm the waterfall shows levers + source tags, the capture bar appears, and the sparkline draws. (No automated test; visual check.)

- [ ] **Step 6: Commit**

```bash
git add docs/workflow-timeline.html
git commit -m "timeline: lever column, source tags, capture + browser tail, history sparkline"
```

---

### Task 10: End-to-end smoke on the live stack + docs

**Files:**
- Modify: `CLAUDE.md`, `STATUS.md`, `WORKFLOW.md` (the measure command notes).

- [ ] **Step 1: Bring the stack up** (CosyVoice WSL -> avatar -> pipeline, per the P15 order) or use the launcher; confirm `:7860` and `:8002` are up and no browser tab is on `/studio`.

- [ ] **Step 2: Run the full harness (probe path first, most robust)**

Run: `python -m scripts.measure --no-browser --turns 3 --offline-capture`
Expected: a MEASURE REPORT with the capture row (may be `unknown` if the synthetic mic VAD-split), all through-stages populated, `[render]` giving the avatar row, and `output/measure_history.jsonl` gaining a line. Confirm `docs/measure_data.js` rewrote.

- [ ] **Step 3: Run the browser path**

Run: `python -m scripts.measure --turns 3`
Expected: Playwright drives `/studio`, `[client-playout]` lines appear in `logs/pipeline.log`, and the waterfall's `Browser jitter buffer` + `Browser decode + playout` rows are populated with `browser-stats`/`browser-audio` sources (not `est`). If Playwright can't launch, it prints the fallback line and still completes via the probe.

- [ ] **Step 4: Verify the fresh/warm split + compare**

Run twice, then: `python -m scripts.measure --compare -2 -1`
Expected: a per-stage delta table between the two runs.

- [ ] **Step 5: Update docs.** In `CLAUDE.md` (the "Avatar A/V test tooling" block) and `STATUS.md`/`WORKFLOW.md`, replace the old `measure.py` description with the new: N-turn median + fresh/warm, real browser playout via Playwright/`?measure=1` beacon, the `[render]`/`[client-playout]` log lines, `--compare`, and `output/measure_history.jsonl`. Note `pipeline/metrics.py` stays untouched.

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md STATUS.md WORKFLOW.md
git commit -m "docs: document the mic-to-ear latency harness (N-turn, real browser, history)"
```

---

## Self-review notes

- **Spec coverage:** §1 stage table -> Tasks 1,2,3 (render),4/5 (browser). §2 both browser paths -> Tasks 4 (auto POST target + human), 5 (studio beacon), 6 (Playwright). §3 N-turn median/fresh-warm -> Tasks 1 (aggregate), 8 (orchestration). §4 history -> Task 7. §5 code structure -> Tasks 1-8. §6 HTML -> Task 9. Testing -> Tasks 1,2,4,7 + Task 10 smoke.
- **Backward-compat:** the existing 7 archive-test cases stay green because `build_waterfall` defaults `capture=None` (t0-relative, unchanged) and the inserted `render`/`jitter` rows are `unknown` when their anchors are absent (telescoping absorbs them). `parse_playout_beacon` is revived, turning the previously-red import green.
- **Type consistency:** anchor keys (`llm_recv, llm_ttfb, tts_recv, tts_ttfb, render, bot_started, client_arrival, jitter, playout, capture`) are identical across `waterfall._WATERFALL_STAGES`, `logparse.build_turn`, and `__main__._turn_anchor_dict`. Beacon field names (`recv, onset, jitterMs, rttMs`) match across Task 4 (endpoint), Task 5 (studio JS), and `logparse.parse_beacon_full`.
- **Page-specific unknowns to resolve during Task 5/6:** the `/studio` connect-button and text-input selectors, and the exact "bot stopped" hook to fire `_postBeacon`. Called out inline in those tasks.
