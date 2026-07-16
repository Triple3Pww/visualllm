"""Unified mic-to-ear latency harness. Drives N turns, parses the pipeline.log delta, measures
the REAL browser output delay (Playwright real Chromium or the /studio ?measure=1 beacon), and
writes one report + docs/measure_data.js + appends output/measure_history.jsonl.

Run (stack up, no browser tab on /studio):
    python -m scripts.measure --turns 5                 # real Chromium (browser E+F), fallback probe
    python -m scripts.measure --no-browser --turns 3    # headless probe: precise capture + arrival
    python -m scripts.measure --no-browser --offline-capture   # + a clean lip offset
    python -m scripts.measure --compare -2 -1           # diff the last two history runs
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from scripts.measure import ANCHOR_KEYS, aggregate_turns, answer_onset_epoch, build_waterfall
from scripts.measure import drive, logparse, report

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")


async def _drive(args):
    """Drive N turns. Prefer the real browser (measures browser E+F); fall back to the probe.
    Returns (path_kind, captures): captures is [] for the browser path (anchors come from the log
    + beacon), else a list of (play_start_epoch, awall) per probe turn."""
    if args.browser:
        ok = await drive.run_browser_turns(args.mic, args.turns, lead=args.blead, tail=args.btail)
        if ok:
            return "browser", []
    captures = []
    for k in range(args.turns):
        print(f"  probe turn {k + 1}/{args.turns}...")
        ps = time.time()
        _vwall, awall, _ct = await drive.run_probe(args.mic, args.lead, args.tail, args.duration)
        captures.append((ps, awall))
    return "probe", captures


def _anchors_for_turn(turn, capture, beacon, onset_rel, jb_est_s):
    """One turn's t0-relative anchors for the waterfall (scalars from the log tuples)."""
    a = dict(
        llm_recv=turn.get("llm_recv"),
        llm_ttfb=turn["llm_ttfb"][0] if turn.get("llm_ttfb") else None,
        tts_recv=turn.get("tts_recv"),
        tts_ttfb=turn["tts_ttfb"][0][0] if turn.get("tts_ttfb") else None,
        render=turn.get("render"),
        bot_started=turn.get("bot_started"),
        capture=capture,
    )
    if beacon:  # browser path: REAL transport arrival + jitter buffer + playout
        recv, jit = beacon.get("recv"), beacon.get("jitter")
        a["client_arrival"] = recv
        a["jitter"] = (recv + jit) if (recv is not None and jit is not None) else None
        a["playout"] = beacon.get("onset")
    else:       # probe path: measured arrival, ESTIMATED playout (arrival + configured jitter ms)
        a["client_arrival"] = onset_rel
        a["jitter"] = None
        a["playout"] = (onset_rel + jb_est_s) if onset_rel is not None else None
    return a


def main():
    ap = argparse.ArgumentParser(description="Mic-to-ear latency harness.")
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--browser", dest="browser", action="store_true", default=True,
                    help="drive a real Chromium for the true browser output delay (default)")
    ap.add_argument("--no-browser", dest="browser", action="store_false",
                    help="use the headless aiortc probe (precise capture + arrival, estimated playout)")
    ap.add_argument("--mic", default="output/_zh_q_def.wav")
    ap.add_argument("--lead", type=float, default=8.0)
    ap.add_argument("--tail", type=float, default=28.0)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--blead", type=float, default=2.0, help="browser fake-mic lead silence")
    ap.add_argument("--btail", type=float, default=6.0, help="browser fake-mic tail silence (turn gap)")
    ap.add_argument("--fps", type=int, default=14)
    ap.add_argument("--offline-capture", action="store_true",
                    help="also drive the MuseTalk server directly for a clean lip offset")
    ap.add_argument("--offline-wav", default="output/reply_concise.wav")
    ap.add_argument("--compare", nargs=2, type=int, metavar=("A", "B"),
                    help="diff two run-history rows by index (e.g. -2 -1)")
    ap.add_argument("--machine", default="this box (RTX 5060 Ti, Blackwell)")
    ap.add_argument("--stack", default="Deepgram/Sherpa STT - OpenRouter LLM - CosyVoice (vLLM/WSL) TTS - MuseTalk avatar")
    args = ap.parse_args()

    if args.compare:
        report.compare_runs(*args.compare)
        return

    print(f"[1/3] driving {args.turns} turns through the live pipeline...")
    path_kind, captures = asyncio.run(_drive(args))

    print("[2/3] parsing pipeline.log for the driven turns...")
    lines = logparse.parse_lines()
    idxs = logparse.find_turn_indices(lines)
    if not idxs:
        raise SystemExit("No [TTFO ...] line in pipeline.log -- did a turn complete? "
                         "(Is the stack up, and did the driver reach it?)")
    n = min(args.turns, len(idxs))
    turns_raw = [logparse.build_turn(lines, i) for i in idxs[-n:]]

    jb_est_s = float(os.getenv("CLIENT_JITTER_BUFFER_MS", "400") or 400) / 1000.0
    sdur = drive.speech_duration(args.mic)   # the wav's real speech length
    anchor_dicts, pm, offline_lip = [], {}, None
    for k, turn in enumerate(turns_raw):
        # Capture = pre-t0 cost, straight from the log: (t0 - 'User started speaking') is the span
        # the VAD perceived; subtract the wav's real speech length and what remains is the VAD
        # stop-hangover + Smart-Turn end-of-turn decision. No wall-clock guessing (a pre-connect
        # time.time() is seconds off because of ICE setup + the greeting), and it works on BOTH
        # the probe and browser paths (each turn logs 'User started/stopped speaking').
        capture, onset_rel = None, None
        us = turn.get("user_started")
        if us is not None:
            cap = round((-us) - sdur, 3)
            capture = cap if cap >= 0 else None
        if path_kind == "probe" and k < len(captures):
            _ps, awall = captures[k]
            onset = answer_onset_epoch(awall, turn["t0_epoch"]) if awall else None
            onset_rel = round(onset - turn["t0_epoch"], 3) if onset is not None else None
        beacon = logparse.parse_beacon_full(lines, turn["t0"])
        anchor_dicts.append(_anchors_for_turn(turn, capture, beacon, onset_rel, jb_est_s))

    # Receiver-side metrics + optional lip offset come from the LAST probe capture (browser -> {}).
    if path_kind == "probe" and captures:
        _ps, last_awall = captures[-1]
        pm = drive.probe_metrics([], last_awall, _ps, args.fps)  # arrival gaps only (no vwall kept)
    if args.offline_capture:
        ow = args.offline_wav if Path(args.offline_wav).exists() else args.mic
        print(f"[3/3] offline avatar capture for a clean lip offset (wav={ow})...")
        offline_lip = asyncio.run(drive.offline_capture(ow, args.fps))
    else:
        print("[3/3] offline capture skipped.")

    keys = ANCHOR_KEYS + ["capture"]
    agg = aggregate_turns(anchor_dicts, keys)
    med = agg["median"]
    rows = build_waterfall(med, playout_source=("browser-audio" if path_kind == "browser" else "est"),
                           capture=med.get("capture"))
    last = turns_raw[-1]
    stage_medians = {r["stage"]: r["cum"] for r in rows if r["status"] == "ok"}
    e2e = next((r["cum"] for r in reversed(rows) if r["status"] == "total"), None)
    hist = [dict(when=h.get("when"), e2e_median=h.get("e2e_median")) for h in report.read_history(30)]

    rep = {
        "meta": {"when": last["t0"].strftime("%Y-%m-%d %H:%M"), "question": last.get("question"),
                 "machine": args.machine, "stack": args.stack, "path": path_kind,
                 "ttfo": last["ttfo_s"], "ttfo_target": 3.0, "ttfo_pass": last["ttfo_pass"],
                 "turns": len(turns_raw), "e2e_median": e2e, "stage_medians": stage_medians,
                 "fresh": agg["fresh"], "warm": agg["warm"], "p95": agg["p95"], "history": hist},
        "events": report.build_events(last),
        "handoffs": report.build_handoffs(last),
        "metrics": report.build_metrics(last, pm, offline_lip),
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
