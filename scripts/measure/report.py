"""Report assembly + output: the timeline events / handoffs / receiver metrics the HTML renders,
the JSON + measure_data.js writers, the run-history JSONL (trend tracking + --compare), and the
console summary. All display; the latency math lives in waterfall.py."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
JSON_OUT = ROOT / "output" / "measure_report.json"
JS_OUT = ROOT / "docs" / "measure_data.js"
HISTORY = ROOT / "output" / "measure_history.jsonl"

RING, DOT = "◯", "●"  # 'receives' ring / 'emits' dot

# The latency-relevant .env knobs snapshotted per run so the history answers "did this change help?".
ENV_KNOBS = ["LANGUAGE", "COSYVOICE_MODEL", "OPENROUTER_PROVIDER_ONLY", "OPENROUTER_MODEL",
             "MUSETALK_LEAD_FRAMES", "MUSETALK_FPS", "MUSETALK_TRT", "COSYVOICE_FIRST_PIECE",
             "COSYVOICE_FIRST_PIECE_ZH", "FILLER_WORDS", "VAD_STOP_SECS", "CLIENT_JITTER_BUFFER_MS"]


# --------------------------------------------------------- timeline events
def build_events(turn):
    """Turn the parsed anchors into the events[] the HTML swimlane renders."""
    ev = []
    us = turn["user_started"] if turn.get("user_started") is not None else -2.0
    sents = turn.get("sentences") or []
    ttfb = turn.get("tts_ttfb") or []
    proc = turn.get("tts_proc") or []
    bs = turn["bot_started"]
    bstop = turn["bot_stopped"] if turn.get("bot_stopped") is not None else bs

    ev.append(dict(stage="capture", t=us, end=0, kind="span", label="User speaking",
                   why="Browser mic -> WebRTC -> Silero VAD listens for the end of the utterance.",
                   src="log user-turn-started"))
    ev.append(dict(stage="stt", t=us, end=0, kind="span", label="STT receives mic (Deepgram/Sherpa/SenseVoice)",
                   why="STT's input is the live mic the whole time the user talks. Deepgram/Sherpa stream partials; "
                       "SenseVoice self-segments (a zipformer endpoint detector) and transcribes the whole buffered "
                       "utterance at end-of-speech -- so its transcript is already in hand when the endpoint fires.",
                   src="log STT stream"))
    ev.append(dict(stage="capture", t=0, kind="turn", label="User STOPPED speaking - t0",
                   why="The STT endpoint (self-seg) or Silero VAD, plus the turn-analyzer, agree the turn ended. "
                       "This instant starts the <3s TTFO stopwatch. For SenseVoice the pre-t0 cost is dominated by "
                       "SENSEVOICE_ENDPOINT_SILENCE (trailing-silence before the endpoint fires).",
                   src="log user-turn-stopped"))
    ev.append(dict(stage="stt", t=0, kind="emit", label="STT emits final transcript",
                   why=(f"\"{turn['question']}\" " if turn.get("question") else "") + "pushed into the LLM context aggregator.",
                   src="log t0"))
    ev.append(dict(stage="llm", t=turn.get("llm_recv") or 0.0, kind="recv", **{"from": "STT"},
                   label=f"{RING} LLM receives the transcript",
                   why="Generation starts here; the LLM was pre-warmed on connect, so no cold start.",
                   src="log 'Generating chat from context'"))
    if turn.get("llm_ttfb"):
        lt, lv = turn["llm_ttfb"]
        ev.append(dict(stage="llm", t=0, end=lt, kind="span", label="OpenRouter generating", subtle=True,
                       why="Streams tokens; the whole answer is ready well before TTS finishes sentence 1.",
                       src="log LLM"))
        ev.append(dict(stage="llm", t=lt, kind="emit", label="LLM emits first token",
                       why=f"OpenRouter TTFB {lv:.3f}s - the cloud hop is the LLM's main cost.",
                       src=f"log LLM TTFB {lv:.3f}s"))

    for i, (st, text) in enumerate(sents):
        done = sents[i + 1][0] if i + 1 < len(sents) else (proc[-1] if proc else st)
        fb = ttfb[i] if i < len(ttfb) else None
        ev.append(dict(stage="tts", t=st, kind="recv", **{"from": "LLM"},
                       label=f"{RING} TTS receives sentence {i+1}",
                       why=(f"\"{text}\" " if i == 0 else "") +
                           ("Starts as the previous sentence finished - CosyVoice synthesizes one sentence at a time (serial)."
                            if i > 0 else "First complete sentence flushed early so TTS can start before the full answer exists."),
                       src="log run_tts"))
        ev.append(dict(stage="tts", t=st, end=done, kind="span", label=f"CosyVoice synthesizing sentence {i+1}",
                       why=(f"First audio at TTFB {fb[1]:.3f}s; " if fb else "") + (f"done at +{done:.2f}s. ") +
                           ("This first chunk is the single biggest piece of TTFO." if i == 0
                            else "Synthesized while earlier audio is already playing, so it doesn't affect TTFO."),
                       src="log run_tts->TTFB"))
        if fb:
            ev.append(dict(stage="tts", t=fb[0], kind="emit", label=f"TTS emits sentence-{i+1} first chunk",
                           why=f"CosyVoice TTFB {fb[1]:.3f}s after receiving the sentence." +
                               (" This is what starts the bot speaking." if i == 0 else ""),
                           src=f"log TTS TTFB {fb[1]:.3f}s"))

    first_tts_fb = turn["tts_ttfb"][0][0] if turn.get("tts_ttfb") else bs
    ev.append(dict(stage="avatar", t=0, end=bs, kind="span", label="Idle / neutral frames", subtle=True,
                   why="A calm neutral face between turns (real-time fps) so the picture is never frozen while TTS works.",
                   src="probe: video pre-speech"))
    ev.append(dict(stage="avatar", t=first_tts_fb, kind="recv", **{"from": "CosyVoice"},
                   label=f"{RING} MuseTalk receives the voice",
                   why="The avatar forwards the first PCM chunk to the :8002 render server the instant TTS emits it (real-time-paced).",
                   src="~ TTS first chunk"))
    if turn.get("render") is not None:
        ev.append(dict(stage="avatar", t=turn["render"], kind="emit", label="MuseTalk first rendered frame",
                       why="First real lip-synced frame back from the render server. The intra-avatar render "
                           "latency (first voice chunk -> this frame) is the [render] log line.",
                       src="log [render] first-frame"))
    ev.append(dict(stage="deliver", t=bs, kind="turn", big=True, label=f"Bot started speaking -> TTFO {turn['ttfo_s']}s",
                   why="The VOICE starts reaching the client here. Under steady (video-master) sync the voice is HELD "
                       "until MUSETALK_LEAD_FRAMES lip frames are rendered, then released in step with them -- so this "
                       f"instant is the synced A/V start. TTFO measures it: {turn['ttfo_s']}s vs the 3s target.",
                   src="log [TTFO] (audio-path event)"))
    ev.append(dict(stage="avatar", t=bs, end=bstop, kind="span", label="MuseTalk lip-sync render",
                   why="Mouth-region frames, steady/video-master sync (default): the voice is paced to the REAL "
                       "rendered frames, so a render stall pauses the voice instead of drifting.",
                   src="log render window"))
    ev.append(dict(stage="deliver", t=bstop, kind="turn", label="Bot stopped speaking - turn complete",
                   why="Full answer delivered. Mic un-mutes; the assistant aggregator records the turn.",
                   src="log bot-stopped"))
    return ev


def build_handoffs(turn):
    sents = turn.get("sentences") or []
    first_tts_fb = turn["tts_ttfb"][0][0] if turn.get("tts_ttfb") else turn["bot_started"]
    s1 = sents[0][0] if sents else 0.0
    return [
        dict(**{"from": "stt"}, to="llm", t=turn.get("llm_recv") or 0.0, what="final transcript (text)",
             note="The question text enters the LLM (generation starts immediately, pre-warmed)."),
        dict(**{"from": "llm"}, to="tts", t=s1, what="sentence 1 (text)",
             note="First complete sentence flushed early, so speech can begin before the full answer exists."),
        dict(**{"from": "tts"}, to="avatar", t=first_tts_fb, what="first voice chunk (16kHz PCM)", star=True,
             note="CosyVoice -> MuseTalk: the avatar forwards the chunk to the :8002 render server the instant TTS emits it."),
        dict(**{"from": "avatar"}, to="deliver", t=turn["bot_started"], what="voice starts (synced A/V start)",
             note="MuseTalk -> browser: under steady the voice is held for the lead cushion, then released paced to "
                  "the real rendered frames. This synced start is TTFO."),
    ]


def build_metrics(turn, pm, offline_lip):
    def tag(cond_ok, cond_warn=False):
        return "ok" if cond_ok else ("warn" if cond_warn else "bad")
    M = [dict(k="TTFO", v=str(turn["ttfo_s"]), u="s", n="target 3s",
              tag="ok" if turn["ttfo_pass"] else "bad")]
    if "startup_s" in pm:
        M.append(dict(k="Startup (connect -> 1st frame)", v=str(pm["startup_s"]), u="s", n="incl. idle warmup", tag=""))
        M.append(dict(k="Received video", v=str(pm["recv_fps"]), u="fps", n="server output rate", tag="ok"))
        M.append(dict(k="Frame interval", v=str(pm["frame_ms_mean"]), u="ms mean",
                      n=f"p95 {pm['frame_ms_p95']} - max {pm['frame_ms_max']}", tag=""))
        M.append(dict(k="Freeze (max gap)", v=str(pm["freeze_ms"]), u="ms",
                      n="OK <500ms" if pm["freeze_ms"] < 500 else "FAIL >500ms",
                      tag=tag(pm["freeze_ms"] < 500)))
    if "audio_gap_max_ms" in pm:
        M.append(dict(k="Audio arrival gap", v=str(pm["audio_gap_max_ms"]), u="ms max",
                      n=f"p95 {pm['audio_gap_p95_ms']}ms",
                      tag=tag(pm["audio_gap_max_ms"] < 50, pm["audio_gap_max_ms"] < 80)))
    lip = offline_lip if offline_lip else (
        dict(ms=pm["lip_offset_ms"], corr=pm.get("lip_offset_corr")) if "lip_offset_ms" in pm else None)
    if lip:
        sign = "lips lag" if lip["ms"] > 0 else "lips lead"
        src = "offline" if offline_lip else "webrtc"
        corr = lip.get("corr")
        if corr is not None and corr < 0.3:
            M.append(dict(k="Lip offset", v=f"{lip['ms']:+d}", u="ms",
                          n=f"{sign}, corr {corr} - LOW-CONF ({src})", tag=""))
        else:
            M.append(dict(k="Lip offset", v=f"{lip['ms']:+d}", u="ms", n=f"{sign}, corr {corr} ({src})",
                          tag=tag(abs(lip["ms"]) < 80, abs(lip["ms"]) < 150)))
    else:
        M.append(dict(k="Lip offset", v="n/a", u="", n=pm.get("lip_offset_note", "unavailable"), tag=""))
    M.append(dict(k="Frames / audio pkts", v=str(pm.get("video_frames", 0)),
                  u=f"/ {pm.get('audio_packets', 0)}", n="over the capture", tag=""))
    return M


# --------------------------------------------------------- output writers
def write_outputs(report):
    JSON_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    js = "// AUTO-GENERATED by scripts/measure -- do not edit by hand.\n"
    js += "window.MEASURE = " + json.dumps(report, ensure_ascii=False) + ";\n"
    JS_OUT.write_text(js, encoding="utf-8")


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(ROOT), text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def append_history(report, env_knobs):
    meta = report["meta"]
    row = dict(when=meta.get("when"), commit=_git_commit(), env=env_knobs,
               turns=meta.get("turns"), e2e_median=meta.get("e2e_median"),
               stage_medians=meta.get("stage_medians", {}))
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_history(n=30):
    if not HISTORY.exists():
        return []
    rows = [json.loads(l) for l in HISTORY.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[-n:]


def compare_runs(a_idx, b_idx):
    rows = read_history(10 ** 9)
    if not rows:
        print("no run history yet (output/measure_history.jsonl is empty).")
        return
    a, b = rows[a_idx], rows[b_idx]
    print(f"compare  A={a['when']} ({a['commit']})   B={b['when']} ({b['commit']})")
    for k in sorted(set(a.get("stage_medians", {})) | set(b.get("stage_medians", {}))):
        av, bv = a["stage_medians"].get(k), b["stage_medians"].get(k)
        if av is None or bv is None:
            print(f"  {k:<40} A={av}  B={bv}")
        else:
            print(f"  {k:<40} {av:+.3f} -> {bv:+.3f}  ({bv - av:+.3f}s)")


# --------------------------------------------------------- console summary
def print_summary(report):
    t = report["meta"]
    print("\n==================== MEASURE REPORT ====================")
    print(f"turns: {t.get('turns')}   last question: \"{t.get('question')}\"  @ {t['when']}")
    print(f"TTFO(last): {t['ttfo']}s (target {t['ttfo_target']}s) {'PASS' if t['ttfo_pass'] else 'FAIL'}")
    print("latency waterfall (median over turns; speech-end -> user hears):")
    for r in report.get("waterfall", []):
        d = f"{r['delta']:+.2f}s" if r["delta"] is not None else "   ?  "
        c = f"{r['cum']:.2f}s" if r["cum"] is not None else "  ?  "
        src = f"[{r['source']}]" if r["source"] else ""
        lev = f"  <- {r['lever']}" if r.get("lever") else ""
        print(f"  {r['stage']:<42} {d:>8}  cum {c:>7}  {src:<15}{lev}")
    fresh, warm = t.get("fresh") or {}, t.get("warm") or {}
    if fresh and warm:
        fe, we = fresh.get("playout") or fresh.get("bot_started"), warm.get("playout") or warm.get("bot_started")
        if fe is not None and we is not None:
            print(f"session degradation (warm - fresh, at last measured stage): {we - fe:+.2f}s")
    print(f"wrote {JSON_OUT}")
    print(f"wrote {JS_OUT}  -> open docs/workflow-timeline.html (auto-uses it)")
    print("=======================================================\n")
