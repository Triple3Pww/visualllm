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


def answer_onset_epoch(samples, t0_epoch, guard=0.15, thresh_frac=0.02, run=3):
    """First SUSTAINED audio frame after t0 = the answer ARRIVING at the client.

    samples: list of (arrival_epoch, rms). Frames at/after t0_epoch+guard are considered, so the
    greeting (well before t0) and inter-turn silence are skipped; the threshold is a fraction of
    the post-t0 peak, and `run` consecutive frames must clear it so a lone spike doesn't trigger.
    Returns the onset epoch (same clock as the log's t0), or None.

    thresh_frac detects PRESENCE (audio vs digital silence), NOT loudness -- this is an arrival
    anchor, and the caller bills everything before it to "Transport + encode + network", a row
    whose physics floor is ~30ms. It was 0.18 (2026-07-16): 18% of the WHOLE reply's peak, so a
    loud passage LATER in the answer raised the bar retroactively and dragged the reported onset
    late. Measured live on one turn: 0.271s reported vs 0.116s of real transport (a 0.154s
    attack-envelope bias), and a reply with loud late dynamics drove the same row to 3.37s -- on
    a loopback hop, i.e. ~100x the floor. The bias was content-dependent, which is why that row
    "varied" 0.27-0.91s across turns and read as network jitter.

    0.02 sits far above the noise floor (inter-turn audio measures a true digital 0.0000: the
    send track emits exact-zero silence when idle) and far below speech level, so it triggers on
    the first real signal instead of waiting out the first word's attack. Kept as a FRACTION, not
    an absolute: the probe scores rms on int16 samples (peak ~2.7e3) while the browser beacon
    scores it normalized (peak ~1.0), and only a fraction ports across both. `run` still rejects
    a lone spike. NOTE: runs recorded before this change measured the old anchor, so that row is
    not comparable across the 2026-07-16 commit -- `--compare` will show a step it did not earn.
    """
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

    anchors: dict of t0-relative offsets (s); a None or NEGATIVE anchor -> 'unknown' row that
    does NOT corrupt the running sum (the next known stage's delta absorbs the gap, so ok-row
    deltas telescope to the last known cum). A negative anchor is a prior-turn artifact (a stage
    can't complete before t0), never a real -X.XXs latency.
    capture: t0 - speech_end (s), the pre-t0 VAD-hangover+turn-end cost, or None. When present it
    is prepended as the first row and added to every downstream cum (so the total is a true
    mic-to-ear sum); when absent the waterfall stays t0-relative, unchanged from before.
    Returns ordered rows {stage, delta, cum, source, lever, status}; a final 'total' row carries
    the end-to-end cum.
    """
    rows = []
    if capture is not None and capture >= 0:
        offset = capture
        rows.append(dict(stage="Capture: speech-end -> t0 (endpoint silence + turn-end)",
                         delta=round(capture, 3), cum=round(capture, 3),
                         source="driver",
                         lever="SENSEVOICE_ENDPOINT_SILENCE (self-seg STT) / VAD_STOP_SECS / turn strategy",
                         status="ok"))
        prev = capture
    else:
        offset = 0.0
        rows.append(dict(stage="Capture: speech-end -> t0 (endpoint silence + turn-end)",
                         delta=None, cum=None, source="driver",
                         lever="SENSEVOICE_ENDPOINT_SILENCE (self-seg STT) / VAD_STOP_SECS / turn strategy",
                         status="unknown"))
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
    """Per-key median + p95 across turns, plus a fresh (turns 0-1) vs warm (turns >=4, or the last
    2 if fewer than 5) split so the session-degradation shows up as an explicit number."""
    def sub(subset):
        return {k: _median([t.get(k) for t in subset]) for k in keys}
    warm_idx = turns[4:] if len(turns) >= 5 else turns[-2:]
    return {
        "median": {k: _median([t.get(k) for t in turns]) for k in keys},
        "p95": {k: _p95([t.get(k) for t in turns]) for k in keys},
        "fresh": sub(turns[:2]),
        "warm": sub(warm_idx),
    }
