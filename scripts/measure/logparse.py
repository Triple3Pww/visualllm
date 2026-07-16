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

    # t0 = 'User stopped speaking' (the TTFO turn-end); 'Generating chat' is the SEPARATE
    # llm_recv anchor (STT finalize + context assembly), so that segment is measured, not
    # collapsed. Fall back to Generating-chat, then bot-start, if the turn-end line is absent.
    t0 = t0_gen = us_start = question = None
    for dt, txt in lines[:bi][::-1]:
        if t0 is None and "User stopped speaking" in txt:
            t0 = dt
        if t0_gen is None and "Generating chat from context" in txt:
            t0_gen = dt
        # This turn's speech start: the last 'User started speaking' at/before t0 (found only
        # once t0 is located so a prior turn's start can't be grabbed). Scanned here, not in the
        # +/-window below, because a long utterance starts well before the window's -3s bound.
        if us_start is None and t0 is not None and "User started speaking" in txt and dt <= t0:
            us_start = dt
        if question is None:
            # The LAST user message in the context (each dict ends '}'); non-greedy per message so
            # a multi-turn accumulated context isn't swallowed whole (the '}]'-anchored regex did).
            qs = re.findall(r"'role': 'user', 'content': '(.*?)'\}", txt)
            if qs:
                question = qs[-1]
        if t0 and t0_gen and us_start and question:
            break
    if t0 is None:
        t0 = t0_gen if t0_gen is not None else bot_started_t

    def off(dt):
        return round((dt - t0).total_seconds(), 3)

    if ttfo_s is None:
        ttfo_s = round((bot_started_t - t0).total_seconds(), 2)
        ttfo_pass = ttfo_s <= target_s

    turn = dict(t0=t0, t0_epoch=t0.timestamp(), ttfo_s=ttfo_s, ttfo_pass=ttfo_pass,
                bot_started=off(bot_started_t), question=question)

    # STT finalize -> LLM start: t0 -> the Generating-chat line (>=0; None if it precedes t0).
    llm_recv = None
    if t0_gen is not None:
        r = off(t0_gen)
        llm_recv = r if r >= 0 else 0.0

    user_started = off(us_start) if us_start is not None else None
    win = [(dt, txt) for dt, txt in lines if -3 <= (dt - t0).total_seconds() <= 60]
    llm_ttfb = render = bot_stopped = None
    sentences, tts_ttfb, tts_proc = [], [], []
    for dt, txt in win:
        if "OpenAILLMService" in txt and "TTFB:" in txt and llm_ttfb is None:
            llm_ttfb = (off(dt), float(re.search(r"TTFB: ([\d.]+)s", txt).group(1)))
        m1 = re.search(r"run_tts:\d+ - CosyVoice TTS \[(.*)\]", txt)
        if m1 and dt >= t0:
            sentences.append((off(dt), m1.group(1)))
        if "CosyVoiceTTSService" in txt and "TTFB:" in txt and dt >= t0:
            tts_ttfb.append((off(dt), float(re.search(r"TTFB: ([\d.]+)s", txt).group(1))))
        if "CosyVoiceTTSService" in txt and "processing time:" in txt and dt >= t0:
            tts_proc.append(off(dt))
        if render is None and "[render] first-frame" in txt and dt >= t0:
            render = off(dt)
        if "Bot stopped speaking based on TTSStoppedFrame" in txt:
            bot_stopped = off(dt)

    turn.update(user_started=user_started,
                llm_recv=llm_recv if llm_recv is not None else 0.0,
                llm_ttfb=llm_ttfb, render=render, bot_stopped=bot_stopped,
                tts_recv=sentences[0][0] if sentences else None,
                tts_ttfb=tts_ttfb, tts_proc=tts_proc, sentences=sentences)
    return turn


def smart_turn_trace(lines, t0):
    """The Smart-Turn end-of-turn verdicts around this turn, as (offset_s_from_t0, verdict).
    Lets a REAL turn show its pre-t0 cost directly: how many times the model said INCOMPLETE
    (kept waiting) before COMPLETE, and when. Clip-independent -- works on human speech."""
    out = []
    for dt, txt in lines:
        d = (dt - t0).total_seconds()
        if -12 <= d <= 1 and "End of Turn result" in txt:
            out.append((round(d, 3), "COMPLETE" if "COMPLETE" in txt else "INCOMPLETE"))
    return out


def select_driven_turns(turns, drive_start, want):
    """The turns THIS run actually drove -- newest `want` of them.

    Why this exists: find_turn_indices scans the WHOLE log (every turn ever run, going back
    days), so taking `idxs[-want:]` silently BORROWS turns from earlier sessions whenever the
    driver registers fewer than `want` [TTFO] lines. The failure is invisible and it backfills
    with STALE, usually WORSE turns (an older session is likelier to be degraded), reported as a
    fresh measurement. Measured 2026-07-16: a driver that produced 4 turns had a 5th backfilled
    from a pipeline process 2 minutes older, and its render (2.20s vs the run's 0.51s) was read
    as a "cold start" that never happened. Worst case: the driver fails against an idle stack and
    the tool reports yesterday's turns with no warning at all.

    drive_start=None (the --observe path) keeps everything -- there the pre-existing turns ARE
    the subject. The caller is expected to say so out loud when fewer than `want` survive.
    """
    if drive_start is not None:
        turns = [t for t in turns if t.get("t0_epoch", 0.0) >= drive_start]
    return turns[-want:] if want else turns


def interrupted_turns(turns):
    """Turns whose reply was still going when the NEXT turn began (`turns` must be time-ordered).

    A turn that speaks to completion logs 'Bot stopped speaking based on TTSStoppedFrame'; one the
    next turn interrupts never does, so a missing bot_stopped is the overlap signature. Overlapped
    turns are NOT measuring the system: the interrupt starves the renderer and inflates the
    Avatar-render and transport rows. Measured 2026-07-16 on this stack: render 0.5s -> 2.17s under
    overlap, deterministic enough to look exactly like the documented session-degradation bug --
    and it was not it, it was the driver's own loop period.

    ONLY turns with a successor can be judged. The FINAL turn of a run routinely lacks a bot-stop
    because the driver disconnected mid-reply -- nothing came after it, so nothing interrupted it.
    Flagging that would fire the warning on every run and train the reader to ignore it.
    (Caveat: a reply longer than logparse's ~60s post-t0 window would also read as 'no bot-stop'.
    At ~50s replies there is headroom, but a slower/wordier model could false-positive here.)
    """
    return [t for t in turns[:-1] if t.get("bot_stopped") is None]


def reply_seconds(turns):
    """Longest reply the bot actually SPOKE, from the turns that finished (None if none did).

    This is the number the browser driver's loop period must clear (--blead + speech + --btail),
    and it is measured rather than assumed because it depends on the model + prompt: on this
    stack replies run ~50s, so the 32s --btail default guarantees every turn interrupts the
    previous one. Interrupted turns are excluded -- their length is the interrupt, not the reply.
    """
    xs = [t["bot_stopped"] - t["bot_started"] for t in turns
          if t.get("bot_stopped") is not None and t.get("bot_started") is not None
          and t["bot_stopped"] > t["bot_started"]]
    return max(xs) if xs else None


def _pick_beacon(lines, target_epoch=None, max_gap=6.0):
    """Pick THIS turn's [client-playout] beacon. Match by the beacon's own `recv` epoch being
    closest to `target_epoch` (the turn's bot-start = when audio should reach the browser) -- this
    survives looped/overlapping browser turns where a broad time window mis-assigns a later turn's
    beacon. target_epoch=None -> the last beacon (the single-turn / test case). None if no beacon is
    within max_gap of the target."""
    cands = []
    for _dt, txt in lines:
        if "[client-playout]" not in txt:
            continue
        m = re.search(r"\[client-playout\]\s*(\{.*\})", txt)
        if not m:
            continue
        try:
            cands.append(json.loads(m.group(1)))
        except ValueError:
            continue
    if not cands:
        return None
    if target_epoch is None:
        return cands[-1]

    def gap(b):
        r = b.get("recv") if b.get("recv") is not None else b.get("onset", b.get("t"))
        return abs(r / 1000.0 - target_epoch) if r is not None else 1e18
    best = min(cands, key=gap)
    return best if gap(best) <= max_gap else None


def parse_playout_beacon(lines, t0, target_epoch=None):
    """Onset offset (s) from this turn's beacon; accepts old {'t':ms} or new {'onset':ms}."""
    b = _pick_beacon(lines, target_epoch)
    if not b:
        return None
    ms = b.get("onset", b.get("t"))
    return None if ms is None else round(ms / 1000.0 - t0.timestamp(), 6)


def parse_beacon_full(lines, t0, target_epoch=None):
    """All beacon fields, t0-relative seconds. jitter/rtt are durations (ms -> s)."""
    b = _pick_beacon(lines, target_epoch)
    if not b:
        return None
    base = t0.timestamp()

    def rel(ms):
        return None if ms is None else round(ms / 1000.0 - base, 6)

    def dur(ms):
        return None if ms is None else round(ms / 1000.0, 6)

    return dict(recv=rel(b.get("recv")), onset=rel(b.get("onset", b.get("t"))),
                jitter=dur(b.get("jitterMs")), rtt=dur(b.get("rttMs")))
