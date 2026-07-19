# Latency matrix — where every millisecond goes, mic to ear

**Measured 2026-07-16, 5 REAL human turns** (spoken into `/studio/?measure=1`, real mic + real
speakers, real browser playout beacon). Regenerate any time with:

```bash
python -m scripts.measure --observe --turns 5      # parse the last N turns YOU just spoke
python -m scripts.measure --turns 6 --btail 58     # or drive them (real Chromium). --btail MUST
                                                   # clear the reply (~50s!) -- see below
python -m scripts.measure --compare -2 -1          # did a change help?
```

> **Three harness traps, all now enforced or flagged in-tool (`docs/PROBLEMS-AND-FIXES.md` P55):**
> **(a) `--btail` must exceed the bot's reply (~50s).** The 32s default does NOT, so every turn interrupts
> the last one and the render + transport rows inflate — this faked a "session degradation" bug for a whole
> session. The tool now detects it and warns.
> **(b) The driver's FIRST turn is lost every run** (`--blead 2` < the ~5s ICE handshake), so you get
> "drove 6, only 5 registered". The tool now says so instead of silently backfilling a stray turn from an
> older session — which is what it used to do, and it corrupted two analyses.
> **(c) `--compare` across the 2026-07-16 commit is invalid for the transport row** (the onset anchor changed
> 0.18 -> 0.02; it shows a ~0.2s probe-path "win" nobody earned).

> **Read this first:** `TTFO` starts its stopwatch at **t0 = "user stopped speaking."** So TTFO only
> ever measures the POST-t0 half. The PRE-t0 half is latency the user sits through that the metric
> structurally cannot see. **Felt delay = pre-t0 + post-t0.** That blind spot is what
> `docs/PROBLEMS-AND-FIXES.md` **P54** was hiding in.

```
you talking ............ [t0] ............ avatar's voice reaches your ear
└──── PRE-t0 ────┘       └───── POST-t0 (what TTFO measures) ─────┘
      ~0.1s                          ~2.91s median
```

## POST-t0 — per stage, per turn (seconds)

| stage | T1 | T2 | T3 | T4 | T5 | **median** | source | lever |
|---|---|---|---|---|---|---|---|---|
| STT finalize -> LLM | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | log | STT provider |
| **LLM first token** | 0.48 | 0.74 | 0.67 | 0.69 | 0.78 | **0.69** | log | `OPENROUTER_PROVIDER_ONLY` / model |
| LLM -> TTS (1st flush) | 0.29 | 0.01 | 0.02 | 0.02 | 0.05 | **0.02** | log | `COSYVOICE_FIRST_PIECE*` |
| **TTS first chunk** | 0.96 | 1.01 | 0.85 | 0.85 | 0.93 | **0.93** | log | first-piece / CUDA graphs / `COSYVOICE_MODEL` / hop |
| **Avatar render** | 0.47 | 0.55 | 0.58 | 0.56 | 0.63 | **0.56** | log | `MUSETALK_TRT` / `MUSETALK_BATCH` |
| steady lead-hold | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | log | `MUSETALK_LEAD_FRAMES` / `MUSETALK_FEED_BURST_S` |
| transport + browser buffering ⚠️ | 0.91 | 0.41 | 0.52 | 0.34 | 0.27 | **0.41** | probe | `CLIENT_JITTER_BUFFER_MS` — **see §Correction** |
| browser jitter buffer | 0.13 | 0.13 | 0.13 | 0.13 | 0.13 | **0.13** | browser-stats | `CLIENT_JITTER_BUFFER_MS` |
| browser decode + playout | 0.06 | 0.09 | 0.12 | 0.00 | 0.06 | **0.06** | browser-audio | device / OS |
| **TOTAL to ear** | **3.30** | **2.96** | **2.91** | **2.60** | **2.85** | **2.91** | | |

> ⚠️ **The transport row is NOT transport** (corrected 2026-07-16 — see §Correction below). Real
> transport measures **~0.13s** and is already at its floor. These numbers are left exactly as the
> harness reported them (they are the record), but the row is mislabelled and its old lever
> (`WEBRTC_VIDEO_BITRATE_MAX`) provably cannot move it. **Do not optimise against this row.**

`T1=HELLO · T2=WHAT THEY CAN · T3=MORE · T4=AGAIN · T5=HELLO`
(T1 is the cold-start turn — its transport spiked to 0.91s; it settles from T2 on.)

## PRE-t0 — per turn

| turn | you spoke | Smart-Turn INCOMPLETE polls | COMPLETE -> t0 (ttfs wait) | TTFO |
|---|---|---|---|---|
| T1 | 1.18s | 0 | 0.11s | 2.19 |
| T2 | 0.85s | 0 | 0.10s | 2.33 |
| T3 | 0.55s | 0 | 0.10s | 2.11 |
| T4 | 3.30s | 0 | **0.00s** | 2.12 |
| T5 | 0.87s | 0 | 0.09s | 2.39 |

**0 INCOMPLETE polls on every real turn** — natural speech endings resolve on the first Smart-Turn
poll. The `COMPLETE -> t0` gap is the **P54 fix** live (was a flat ~1.0s; now = the declared
`ttfs_p99_latency=0.1`). T4 fired instantly (transcript already final).

## What to attack next (ranked by measured cost)

1. **TTS first chunk — 0.93s, but ~70% is a FIXED FLOOR (P56, 2026-07-16). Reframed: NOT "real headroom".**
   Measured isolated on the live server: TTFB = **0.648s + 25.9 ms/char** (zh) — ~0.65s is fixed startup paid
   on ANY input, even 1 char; only ~0.28s is length-dependent and the live first piece is already ~10 chars,
   so the realistic remaining win for the whole §3 lever family is **~0.13s**. **§3.1 is ANSWERED: the English
   split fires** (6/7 openers at the live 18/32; the 7th is a complete short sentence). Mechanism CORRECTED:
   TTFB tracks the first *segment*, not the sentence — CosyVoice's frontend MERGES short text up to an 80-token
   cap (`frontend.py` `token_min_n=60/max=80`), and `COSYVOICE_FIRST_PIECE` wins by DENYING that merge, not by
   shortening a prefill. vLLM-Omni is CLOSED (handoff §3.2, retested). Full plan + do-not-re-open list:
   **`docs/TTS-FIRST-CHUNK-HANDOFF.md`**; floor + residue + mechanism: `docs/PROBLEMS-AND-FIXES.md` **P56**.
   **Higher-value lead surfaced by P56: barge-in leaks TTS residue onto the shared GPU** (an abandoned stream
   keeps synthesizing the whole ~50s reply — +1.0s on the next request; fits the ~1-in-7 +1.7s spike below).
2. **LLM first token — 0.69s.** Already Groq-pinned (`OPENROUTER_PROVIDER_ONLY=Groq`, P21) which killed the
   7-8s tail. Remaining ~0.7s is mostly the cloud hop; a local Ollama would trade it for GPU contention on
   an already-shared card. Low ceiling.
3. **Avatar render — 0.56s.** Already TRT (P16). `MUSETALK_BATCH` is the untried knob; the structural fix is
   a dedicated avatar GPU. **Verified healthy 2026-07-16**: flat 0.44-0.52s across 6 non-overlapping turns,
   and barge-in costs it NOTHING (5/5 interrupted turns 359-532ms = identical to clean, first PCM on the wire
   in +0ms). The "degrades after ~3 turns" repro was a **harness artifact** — the driver's loop period was
   shorter than the 50s reply, so every turn interrupted the last and starved the renderer (P55). Residual,
   open: an intermittent **~1-in-7 +1.7s** first-frame spike, independent of the interrupt path.
4. **Transport — CLOSED, nothing to win (2026-07-16).** Real transport is **~0.13s** and sits at its
   floor; the 0.41s row is mostly artifact + browser-side buffering (§Correction). The pipecat send
   path was *measured* clean (`MEASURE_SEND_TRACE=1`: first chunk queued->on-wire in **0-39ms**, the
   event loop never >20ms late) despite that one loop also carrying RTP + the MuseTalk pump. The only
   real lever on the browser-side remainder is `CLIENT_JITTER_BUFFER_MS` (already 150ms) — a deliberate
   WAN-smoothness purchase; cut it only for a LAN-only viewer and expect choppiness over Tailscale/WAN.
5. **Dead rows — do not bother.** STT->LLM and steady lead-hold both measure 0.00s. The lead-hold row is
   **structurally** 0.00: `bot_started` and `render` resolve to the same instant, so the 14-frame lead's
   real cost is absorbed into the Avatar-render row above it, never shown on its own.

## Correction (2026-07-16): the transport row invented ~0.2s that was never there

The row is a **residual**, not a measurement: `scripts/measure/__main__.py` computes it as
`(recv - jitter) - bot_started`, so every unmodelled delay and every anchor error in that span lands
in it. Its physics floor is ~0.03s (a 10ms send-track tick + 20ms Opus packetization + the hop).
0.41s is ~13x that. Two measurements decomposed it:

| what | measured | how |
|---|---|---|
| pipecat send path (queued -> on the wire) | **0-39ms**, loop never >20ms late | `MEASURE_SEND_TRACE=1`, n=7 |
| **real transport** (`bot_started` -> first audio at a same-box receiver) | **0.116s** | probe, n=1 |
| harness's reported onset, same turn | 0.271s | probe, n=1 |
| **-> pure detector bias** | **0.154s** | difference |

**Root cause:** `answer_onset_epoch` thresholded at **18% of the whole reply's peak**, so a reply that
got loud *later* raised the bar retroactively and dragged the reported onset hundreds of ms late — time
billed to "the network". Being content-dependent, it also explains the row's 0.27-0.91s "jitter" (a
loopback hop cannot vary like that) and a **3.37s** reading observed on one run. Fixed by anchoring on
audio **presence** (`thresh_frac` 0.18 -> 0.02); the probe row fell **0.33s -> 0.14s** on live turns,
matching the 0.116s hand-measurement. Regression test:
`archive/_measure_waterfall_test.py::test_onset_is_not_dragged_late_by_a_loud_later_passage` (the old
square-wave tests stepped 0.0 -> 0.6 and cleared *any* fraction of the peak, which is how this survived).

**Scope — what the fix does and does not touch.** It corrects the **probe** path only. The 5-turn table
above is the **browser** path, whose `recv` comes from the beacon (`rms > 0.01` at rAF granularity),
not from `answer_onset_epoch` — so those numbers are unchanged. Against the measured 0.13s of real
transport, the browser row's remaining ~0.28s is browser-side: NetEq beyond the average
`jitterBufferDelay` that gets subtracted, rAF granularity (**Chrome throttles rAF when the tab is
occluded** — i.e. whenever you speak and then look at the terminal), the analyser's ~43ms RMS window,
and that 0.01 threshold's own attack bias. Some of that is real and felt; some is instrument. It is
**not** transport, and no network lever reaches it.

**Two more rows that are not what they claim:**
- **"Browser decode + playout" is not measured on the probe path** — `__main__.py:68` adds
  `CLIENT_JITTER_BUFFER_MS/1000` as a constant. The 0.15s it prints on a `--no-browser` run is the
  `.env` value echoed back in a column labelled as measurement. Only the browser path measures it.
- The old lever `WEBRTC_VIDEO_BITRATE_MAX` caps the **VP8 video** encoder (`main.py::_configure_webrtc_video_bitrate`).
  The voice is a **separate Opus track**. On a same-box link it has no path to audio arrival — turning it
  would have produced a null result forever.

**The lesson, and it is P54's in reverse.** There, a row *hid* a real dead second. Here, a row
*invented* a quarter-second. Same root cause both times: a number nobody had decomposed. The house rule
"the probe passes what the eye rejects" (P19) has a third face — **a probe can also fail what the wire
already delivered.** Before optimising any row, check it against its physics floor; 13x over budget is
the instrument, not the system.

> **History warning:** rows in `output/measure_history.jsonl` recorded before the 2026-07-16 commit used
> the 0.18 anchor, so `--compare` across that boundary shows a ~0.2s transport "win" on the probe path
> that nobody earned. The per-row git commit identifies which side of the change a run is on.

## Method notes (so the next reader does not get fooled)

- **The probe lies in BOTH directions** (P19/P33/P54). The synthetic clip `_zh_q_def.wav` has a 0.74s comma
  pause, which made Smart Turn poll INCOMPLETE twice and INVENTED a ~1.6s pre-t0 cost that **real speech does
  not have**. Judge pre-t0 from `--observe` on real turns, never from the synthetic mic. **It lies POST-t0
  too** — the transport row invented ~0.2s of network that was never there (§Correction). Both lies were
  caught the same way: comparing a row against its physics floor.
- **A residual row is a suspect, not a measurement.** Transport, and any row derived as the gap between two
  anchors, silently absorbs every unmodelled delay AND every anchor error between them. When such a row is
  far over its floor, decompose it before believing it: `--no-browser` (probe) vs the browser path splits
  browser-side from upstream for free, and `MEASURE_SEND_TRACE=1` splits the pipecat send path from the wire.
- **`--observe` leaves the Capture row blank on purpose**: human speech length is unknown, so the tool
  refuses to fake it and prints the Smart-Turn verdict trace instead (clip-independent, honest).
- **No session degradation in this sample** (warm - fresh = -0.27s; turns got *faster*). The known
  degradation bug needs longer replies than these 5 short turns to reproduce.
- `pipeline/metrics.py` (the TtfoMeter) is **untouched** — the whole waterfall is derived in `scripts/measure/`.

> Companions: `docs/workflow-timeline.html` (the same run, rendered + hoverable) ·
> `docs/PROBLEMS-AND-FIXES.md` **P54** (the pre-t0 second) · `STATUS.md` (current state) ·
> `output/measure_history.jsonl` (every run, with its git commit + `.env` knobs).
