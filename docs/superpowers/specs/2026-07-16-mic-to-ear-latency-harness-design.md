# Mic-to-ear latency harness — design

**Date:** 2026-07-16
**Status:** approved, pre-implementation
**Goal:** Remake the measure system so it efficiently measures the *detailed* per-stage
latency from the microphone to the user's ear — **including the real browser output delay** —
broken down as finely as possible, so the numbers can be used to minimize overall latency
in future changes.

## Why the current system is insufficient

`scripts/measure.py` today drives one turn over a headless aiortc probe, parses `pipeline.log`,
and produces a 7-row waterfall. Three gaps:

1. **Browser output delay is estimated, not measured.** The last row is
   `headless_arrival + CLIENT_JITTER_BUFFER_MS`. The real-browser beacon was removed 2026-07-14
   (it only patched the prebuilt `/client`, unsupported under `MUSETALK_SPLIT=1`). No real number today.
2. **The whole mic → t0 capture segment is invisible.** t0 *is* the turn-end, so VAD-stop hangover +
   Smart-Turn end-of-turn decision never appear — "latency the user feels but the metric can't see."
3. **Avatar intra-server render latency is not logged** — bundled inside the steady lead-hold.

## Decisions (locked with the user)

- **Browser delay fidelity:** measure it for real via **BOTH** an automated real Chromium (Playwright)
  and an instrumented `/studio` beacon for real human sessions.
- **Mic-side scope:** measure the pre-t0 capture segment too (speech-silence → t0).
- **Sampling:** drive **N turns** per run; report per-stage **median + p95** and a **fresh (turns 1-2)
  vs warm (turns >=5) split** so the session-degradation bug is a first-class number.
- **History:** append each run to a history file for "did this change help?" trend tracking.

## The measurement model — stages, source, lever

The waterfall becomes ~11 sub-stages across 6 segments. Each row carries a **source tag**
(`log` / `probe` / `browser-stats` / `browser-audio` / `est`) and the **`.env` lever** that moves it.
Any anchor missing this run renders `unknown` and does NOT corrupt the running sum (the next known
stage's delta absorbs the gap — same telescoping rule as today).

| # | Segment -> stage | Measured from | Lever |
|---|---|---|---|
| A | Capture: speech-silence -> t0 (VAD hangover + Smart-Turn end-of-turn) | driver's known wav speech-end epoch; log "User stopped speaking" = t0 | `VAD_STOP_SECS`, turn strategy |
| B1 | STT: t0 -> LLM starts (final transcript + context assembly) | log t0 -> "Generating chat from context" | STT provider |
| B2 | LLM: generation -> first token (TTFB) | log LLM TTFB | `OPENROUTER_PROVIDER_ONLY`, model |
| B3 | LLM->TTS: first token -> sentence-1 flush | log run_tts | `COSYVOICE_FIRST_PIECE*` |
| C1 | TTS: sentence-1 received -> first audio chunk (TTFB) | log TTS TTFB | first-piece, CUDA graphs, `COSYVOICE_MODEL`, hop |
| D1 | Avatar: first voice chunk -> first rendered frame (intra-server render) | *new* one-line log in `musetalk_video` | `MUSETALK_TRT`, `MUSETALK_BATCH` |
| D2 | Avatar: render -> bot-start (steady lead-hold release) = TTFO end | log `[TTFO]` | `MUSETALK_LEAD_FRAMES`, `MUSETALK_FEED_BURST_S` |
| E | Transport: bot-start -> first RTP at browser (encode + network) | browser receive epoch - t0 (or headless arrival) | `WEBRTC_VIDEO_BITRATE_MAX`, network |
| F1 | Browser: receive -> jitter-buffer emit | browser `getStats` `jitterBufferDelay` | `CLIENT_JITTER_BUFFER_MS` |
| F2 | Browser: jitter emit -> actual sound out (decode + device playout) | browser WebAudio analyser onset | device/OS |
| = | END — user hears | sum A..F | — |

**Honesty rules preserved:** negative anchors (logged before t0) are also `unknown` (prior-turn
artifact), and low-confidence signals are flagged rather than presented as trustworthy — mirroring the
existing lip-offset `corr < 0.3` handling.

## Two browser paths for E/F

**Auto (default one command).** A real **headless Chromium via Playwright**. The wav is the mic via
Chrome flags `--use-fake-device-for-media-stream --use-file-for-fake-audio-capture=<wav>`
(`--autoplay-policy=no-user-gesture-required`). The page taps the decoded inbound stream with a
WebAudio `AnalyserNode` for the **true first-sound instant** (F2), and reads `getStats()` for
`jitterBufferDelay`/`jitterBufferEmittedCount` (F1) and candidate-pair RTT (E). Same box, so the
browser's `Date.now()` stitches to the pipeline log clock exactly as the headless probe already does
(`t0.timestamp()` epoch equality).

**Human.** A `?measure=1`-gated beacon added to the static `/studio` page (reusing the existing
`AudioContext` + `createMediaStreamSource` + `ontrack`). It records the same onset + `getStats`, and
`POST`s a small JSON to a new `POST /client/measure-beacon` (mirrors `/client/say` in the existing
`_inject_client_patches` middleware). `measure.py` reads the latest beacon via `GET /client/measure-beacon`.
Remote/WAN clock offset is resolved by an **NTP-style min-RTT handshake** inside the beacon (server epoch
in the GET response, client keeps the min-RTT sample); absolute remote numbers are flagged `est`, but the
jitter-buffer/decode split is self-contained and stays exact.

**Fallback.** If Playwright is not installed, the harness falls back to today's headless probe + the
jitter-buffer estimate for F, and prints an install hint. It never hard-fails.

## Multi-turn and history

- **N turns (default 5)** driven in one session. Per stage: median + p95. Plus a **fresh vs warm** split
  (turns 1-2 vs >=5) surfaced as an explicit degradation delta.
- **Run history:** append one row per run to `output/measure_history.jsonl` —
  `{when, git_commit, env_knobs{...}, stage_medians{...}, e2e_median}`. A `--compare <A> <B>` mode diffs
  two runs. The HTML shows a small end-to-end sparkline over recent runs.

## Code structure

Refactor `scripts/measure.py` (public entry `python -m scripts.measure` and all output paths unchanged)
into a focused package so each unit has one purpose:

- `scripts/measure/logparse.py` — pipeline.log -> per-turn anchors (from today's `parse_turn`/`_build_turn`).
- `scripts/measure/drive.py` — the two drivers: headless `probe` (reuses `_webrtc_probe`) + `browser` (Playwright).
- `scripts/measure/waterfall.py` — the stage table above + median/p95 + fresh/warm aggregation.
- `scripts/measure/report.py` — JSON / `measure_data.js` / history writers + console summary.
- `scripts/measure/__main__.py` — CLI + orchestration (drives N turns, stitches clocks, assembles the report).

**New instrumentation (all additive, isolated):**
- One `[render] first-frame +Xms` log line in `local_services/musetalk_video.py` (first real rendered
  frame after each turn start) for D1.
- `POST`/`GET /client/measure-beacon` in `pipeline/main.py`'s existing `_inject_client_patches` middleware,
  plus a `measureBeacon`/`serverEpoch` field in the `GET /client/ice-config` payload (like `jitterBufferMs`).
- A `?measure=1` beacon block in `local_services/studio_client/index.html`.
- **`pipeline/metrics.py` (TtfoMeter) stays untouched** — segment A is derived from the wav speech-end +
  the existing "User stopped speaking" log line; no meter change.

## The HTML

Extend `docs/workflow-timeline.html` (same aesthetic, `measure_data.js` stays the single data source):
- Waterfall section gains the **lever column** and **source tags**.
- Swimlane gains the pre-t0 **capture** bar and the **browser-output** tail (E/F1/F2).
- A small **run-history sparkline** of end-to-end latency at the top.
- Median/p95 shown per stage; fresh-vs-warm degradation callout.

## Testing

- `logparse.py`: unit test against a checked-in `pipeline.log` fixture snippet -> known anchors.
- `waterfall.py`: unit tests for telescoping over `unknown`/negative anchors, median/p95, fresh/warm split.
- Beacon endpoint: a small test POSTing a synthetic beacon and reading it back.
- End-to-end smoke: run `python -m scripts.measure` against the live stack (headless probe path) and
  confirm a complete report with the new segments; run the Playwright path when available.

## Non-goals (YAGNI)

- No per-frame render trace from the avatar server (one first-frame anchor is enough for the waterfall).
- No full trend dashboard beyond the sparkline + `--compare`.
- No change to TTFO's t0 definition or to `TtfoMeter`.

## Dependencies / risks

- **Playwright + a Chromium download** is a new dependency in the system (pipeline) Python. Handled by the
  graceful fallback if absent.
- The avatar server is single-client, so turns (probe or Playwright) run sequentially — unchanged from today.
