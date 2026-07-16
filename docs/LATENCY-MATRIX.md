# Latency matrix — where every millisecond goes, mic to ear

**Measured 2026-07-16, 5 REAL human turns** (spoken into `/studio/?measure=1`, real mic + real
speakers, real browser playout beacon). Regenerate any time with:

```bash
python -m scripts.measure --observe --turns 5    # parse the last N turns YOU just spoke
python -m scripts.measure --turns 5              # or drive them automatically (real Chromium)
python -m scripts.measure --compare -2 -1        # did a change help?
```

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
| transport + encode + net | 0.91 | 0.41 | 0.52 | 0.34 | 0.27 | **0.41** | probe | `WEBRTC_VIDEO_BITRATE_MAX` / network |
| browser jitter buffer | 0.13 | 0.13 | 0.13 | 0.13 | 0.13 | **0.13** | browser-stats | `CLIENT_JITTER_BUFFER_MS` |
| browser decode + playout | 0.06 | 0.09 | 0.12 | 0.00 | 0.06 | **0.06** | browser-audio | device / OS |
| **TOTAL to ear** | **3.30** | **2.96** | **2.91** | **2.60** | **2.85** | **2.91** | | |

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

1. **TTS first chunk — 0.93s.** The biggest single row. CosyVoice's first-chunk TTFB scales with the
   INPUT sentence length (it prefills the whole sentence before the first audio token). `COSYVOICE_FIRST_PIECE`
   exists to cut exactly this by flushing a short opening clause first — **worth confirming it actually fires
   on English turns** (the zh path needed its own `_ZH` splitter because the en splitter keys on ASCII
   comma/space; verify the en path is live and its 18/32 MIN/MAX are still the sweet spot).
2. **LLM first token — 0.69s.** Already Groq-pinned (`OPENROUTER_PROVIDER_ONLY=Groq`, P21) which killed the
   7-8s tail. Remaining ~0.7s is mostly the cloud hop; a local Ollama would trade it for GPU contention on
   an already-shared card. Low ceiling.
3. **Avatar render — 0.56s.** Already TRT (P16). `MUSETALK_BATCH` is the untried knob; the structural fix is
   a dedicated avatar GPU.
4. **Transport 0.41s / jitter 0.13s.** `CLIENT_JITTER_BUFFER_MS` is a deliberate WAN-smoothness purchase —
   only cut it for a LAN-only viewer, and expect choppiness over Tailscale/WAN.
5. **Dead rows — do not bother.** STT->LLM and steady lead-hold both measure 0.00s.

## Method notes (so the next reader does not get fooled)

- **The probe lies in BOTH directions** (P19/P33/P54). The synthetic clip `_zh_q_def.wav` has a 0.74s comma
  pause, which made Smart Turn poll INCOMPLETE twice and INVENTED a ~1.6s pre-t0 cost that **real speech does
  not have**. Judge pre-t0 from `--observe` on real turns, never from the synthetic mic.
- **`--observe` leaves the Capture row blank on purpose**: human speech length is unknown, so the tool
  refuses to fake it and prints the Smart-Turn verdict trace instead (clip-independent, honest).
- **No session degradation in this sample** (warm - fresh = -0.27s; turns got *faster*). The known
  degradation bug needs longer replies than these 5 short turns to reproduce.
- `pipeline/metrics.py` (the TtfoMeter) is **untouched** — the whole waterfall is derived in `scripts/measure/`.

> Companions: `docs/workflow-timeline.html` (the same run, rendered + hoverable) ·
> `docs/PROBLEMS-AND-FIXES.md` **P54** (the pre-t0 second) · `STATUS.md` (current state) ·
> `output/measure_history.jsonl` (every run, with its git commit + `.env` knobs).
