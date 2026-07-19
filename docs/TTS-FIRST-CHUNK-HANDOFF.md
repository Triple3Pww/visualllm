# Handoff — TTS first chunk (0.93s), the last big row

**Written 2026-07-16 (28th session).** Everything else in the mic-to-ear waterfall is at its known
limit or closed. This is the one row left with real headroom, and it costs **every turn**.

> **UPDATE 2026-07-16 (29th session, P56) — this framing is now KNOWN-OVERSTATED. Read first.**
> Measured the row's physics floor (the §4 rule this handoff itself preaches, applied to the target):
> TTFB = **0.648s + 25.9 ms/char** (zh, isolated). **~0.65s is a FIXED FLOOR** no text lever touches;
> only ~0.28s is length-dependent, and the live first piece is already ~10 chars → realistic remaining
> win for the whole §3 family is **~0.13s**, not "real headroom". **§3.1 is ANSWERED (the en split fires,
> 6/7).** The stated mechanism ("prefills the whole sentence") is **wrong** — TTFB tracks the first
> *segment*, and CosyVoice MERGES short text up to an 80-token cap; FIRST_PIECE wins by denying the merge.
> **The higher-value lead is now barge-in residue** (an abandoned TTS stream keeps synthesizing the whole
> ~50s reply on the shared GPU → +1.0s on the next request). Full write-up: `docs/PROBLEMS-AND-FIXES.md`
> **P56**. The rest of this doc is preserved as the 28th-session record.

> **Read `docs/LATENCY-MATRIX.md` §Correction and `docs/PROBLEMS-AND-FIXES.md` P55 BEFORE measuring
> anything here.** A full session was just burned on rows that turned out to be instrument artifacts.
> The discipline in §4 below is not ceremony — it is what caught all five of them.

## 1. The target

| row | median | status |
|---|---|---|
| **TTS first chunk** | **0.93s** (real turns) / 0.97s (6-turn browser) | **THE TARGET** |
| LLM first token | 0.69s | Groq-pinned (P21). Mostly the cloud hop. Low ceiling. |
| Avatar render | 0.49-0.56s | TRT (P16). Flat + healthy — verified 2026-07-16, no degradation. |
| transport | ~0.13s real | **CLOSED** (P55). At its floor. Do not re-open. |
| STT->LLM · lead-hold | 0.00s | dead rows |

POST-t0 median is **2.80-2.91s**; pre-t0 adds ~0.1s since P54. **Felt delay ~2.9s.** TTS is ~1/3 of it.

## 2. Why it costs what it costs

**CosyVoice's first-chunk TTFB scales with the INPUT sentence length** — it prefills the whole sentence
before emitting the first audio token. A 16-word opener costs ~3.0s; a short one ~1.7s. Every lever below
exists to attack that one fact.

Live config (verified in `.env`, 2026-07-16): `LANGUAGE=zh`, `COSYVOICE_MODEL=v2` (**not v3, despite
CLAUDE.md's "v3 = baseline" line**), CUDA graphs ON (correct for every language incl. zh — P33 is REVERSED),
`COSYVOICE_FIRST_PIECE=1` (18/32), `COSYVOICE_FIRST_PIECE_ZH=1` (min 5 CJK), `COSYVOICE_FIRST_HOP_ZH=0`,
`COSYVOICE_T2S=1`, `COSYVOICE_TRIM_LEAD=1`, `COSYVOICE_PACE_RATE` unset (= OFF; pacing never delays the
first chunk either way, so it is NOT a lever in either direction).

## 3. The leads, cheapest first

> **Status 2026-07-16: §3.2 (vLLM-Omni) is DEAD — tested and refuted, do not re-open.** That leaves
> **§3.1 the cheapest LIVE lead, and it has still never been checked.** Start there.

### 3.1 Confirm `COSYVOICE_FIRST_PIECE` fires on ENGLISH — ✅ ANSWERED 2026-07-16 (P56): IT FIRES.

Pure text-logic probe at the live 18/32 (no GPU needed — the aggregator is just text): the split fired on
**6/7** realistic openers; the 7th was `"Yes."`, a complete short sentence with nothing to split. The en path
is NOT silently inert the way the zh path once was. **Caveat found:** the MAX cap only guarantees it never cuts
mid-*word*, not mid-*phrase* — it overshoots to the next space, so `"Machine learning is a subset of artificial"`
splits `artificial | intelligence`. That is a prosody/ear question, not a firing question; 18/32 was NOT
re-swept by ear (the ~0.13s at stake makes it low-value). `local_services/first_piece_aggregator.py`.

### 3.2 vLLM-Omni via the REGISTERED-VOICE cache — ❌ DEAD, TESTED 2026-07-16. Do not re-open.

**This was billed as "the strongest lead" and it is refuted.** The 2026-07-15 spike measured vllm-omni at
1.319s vs our 1.035s and closed it as "not a win — per-request ref re-processing is structural". A source
review then found `SpeakerEmbeddingCache` (PR #2630) merged upstream, keyed on a *registered voice name*, and
argued the close only held for the spike's INLINE `ref_audio` path — so ~0.83s FLAT should be reachable by
registering the voice. **It is not.** Measured with the clean same-build A/B the original spike never ran:

| arm — vllm-omni main (0.25.0rc2.dev26) + vllm 0.25.1, same 8 openers, same 9.2s leo ref | median | stddev |
|---|---|---|
| inline ref (control) | **1.648s** | 0.109 |
| **registered voice (the cache path)** | **1.664s** | 0.058 |
| registered voice, **2.5s** truncated ref | **0.630s** | 0.049 |

- **The cache buys NOTHING** — 1.648 vs 1.664 is noise. Not a small win; zero.
- **The 2.5s row proves WHY: ref length still dominates the cached path.** If the cache were hitting, the
  clip's length could not matter — the extraction would be skipped. It is wired in (`cosyvoice3.py:291-332`),
  the voice registers (`{"voices":["leo"]}`), the cache initialises (`Speaker cache ready`) — and does nothing.
- **The engine REGRESSED:** inline was 1.319s on 0.24.0, 1.648s on 0.25.1 (+0.33s). vllm-omni is now
  ~0.6s BEHIND our server, further than when it was first closed.
- Both spikes ran `enforce_eager` (verified in both server logs), so eager is controlled, not a confound.

**Do not read the 0.630s as a lever for us.** That gap is omni re-extracting the reference per request
(s3tokenizer/campplus scales with clip length) — work our server pays **zero** times, since it precomputes the
zero-shot prompt once at startup. What ref length costs US is only the ~230 extra speech tokens in the prefill:
plausibly tens of ms, not a second. Cheap to measure (truncate `COSYVOICE_PROMPT_WAV`, relaunch the WSL TTS
server, run `_ttfb_variance.py`) but the theory says small, and **a shorter ref trades voice similarity — an
EAR question (the user's), not a probe's.**

**Env gotchas, now 5 (3 were pre-paid by the 2026-07-15 checklist, 2 are new — see `output/_start_omni.sh`):**
`s3tokenizer` pip pkg; uv venv MUST use managed python (system 3.12 has no `Python.h`); `CC`/`CXX` = cosyvllm
conda gcc (triton JIT-compiles its launcher stub; WSL has no system gcc); model on WSL ext4 (`torch.load` from
`/mnt/e` = 9P Errno 5); H100-default chunk params regress on a 5060 Ti. **NEW:** the released vllm-omni
(0.24.0, still the latest on PyPI) does **not** contain the cache at all — only unreleased `main` does, and
main needs vllm 0.25.x (`ModuleNotFoundError: vllm.entrypoints.scale_out`). Upgrading vllm re-installs
**flashinfer**, which JIT-compiles and dies on `Could not find nvcc`; uninstalling it then breaks vllm 0.25.1's
*unguarded* import in `flashinfer_sampler_supported()`. The only exit is **`VLLM_USE_FLASHINFER_SAMPLER=0`**,
which short-circuits before that import (`topk_topp_sampler.py:38`). 0.24.0 tolerated a plain uninstall; 0.25.1
does not. **If a future vllm-omni RELEASE ships the cache, the only thing worth re-testing is whether the cache
actually hits** — check the 2.5s-vs-9.2s ref delta first; if length still matters, it does not, and stop there.

### 3.3 `COSYVOICE_MODEL=v2 -> v3` (already built, one relaunch)

v3 = Fun-CosyVoice3-0.5B + flow-TRT + CUDA graphs; the A/B measured a +0.066s gate.
`.env` runs **v2** today. Switch via the config panel's **CosyVoice model** card — a plain `.env` edit needs
the WSL TTS server relaunched, which the pipeline-only Restart does NOT do.
Plan: `docs/superpowers/plans/2026-07-10-cosyvoice3-ab.md`.

### 3.4 Structural, parked by the user: eager end-of-turn

Overlap turn-confirmation with LLM prefill (speculative prefill on STT interims). Still the strongest
untouched lever, but it is **pre-t0** — TTFO structurally cannot see it, so judge it with `--observe` on real
speech only. Explained and parked in the ttfo-sub-1.5s research doc.

## 4. How to measure this without fooling yourself

**Five confident hypotheses died this session. Every one looked like a real defect.** What killed them:

1. **Check the physics floor FIRST.** The transport row was 13x over budget — that alone was the answer,
   before any investigation. If a number is impossible, it is the instrument.
2. **A residual row is a suspect, not a measurement.** Anything derived as the gap between two anchors
   absorbs every unmodelled delay AND every anchor error between them.
3. **The probe lies in BOTH directions.** Pre-t0: the synthetic clip's comma pause invents ~1.6s that real
   speech does not have (P54). Post-t0: the onset detector invented ~0.2s of network (P55).
4. **Judge model/voice quality with an ISOLATED probe**, never the live waterfall.
5. **The live eye is the arbiter, in both directions** (P19/P33). A measured delta is not automatically a
   perceived one — and a probe can fail what the wire already delivered.

**Harness usage that is now enforced (P55 fixes) — but know why:**

```bash
python -m scripts.measure --turns 6 --btail 58   # btail MUST clear the reply (~50s!) or every
                                                 # turn interrupts the last one and the render
                                                 # + transport rows inflate. The tool now warns.
python -m scripts.measure --observe --turns 5    # the ONLY honest pre-t0 (real speech you spoke)
python -m scripts.measure --compare -2 -1        # did it help?
```

- **Known-broken:** `--blead 2` is shorter than the ~5s ICE handshake, so the driver's **first turn is lost
  every run** ("drove 6, only 5 registered"). The tool now says so instead of silently backfilling a stray
  from an older session. Fix this before trusting turn counts.
- **`--compare` across the 2026-07-16 commit is invalid for the transport row** — the onset anchor changed
  (0.18 -> 0.02), so it shows a ~0.2s probe-path "win" nobody earned.
- **Headless Chromium has no audio device**, so the "browser decode + playout" row reads 0.00 on the
  Playwright path; only a real browser measures it.
- **Diagnostics available, both default-OFF/zero-cost:** `MEASURE_SEND_TRACE=1` (pipecat send path:
  queued -> on the wire + loop lateness), `MUSETALK_PROFILE=1` (per-8-frame-segment feat/whisper/gpu/composite
  — this is what proved the render spike was NOT compute). `[barge]` logs are always on (3/turn).

## 5. Do NOT re-open (each cost a session, all closed by measurement)

- **transport / network** — ~0.13s, at floor (P55). `WEBRTC_VIDEO_BITRATE_MAX` is video-only; it cannot move it.
- **Event-loop contention on the audio send path** — measured clean: 0-39ms, loop never >20ms late (P55).
- **Browser jitter buffer as the transport cost** — the no-browser probe read the same number (P55).
- **`MUSETALK_LEAD_FRAMES` below 14** — the user's live eye rejected EVERY value below 14, twice (P19/P22).
- **CUDA graphs off for zh** — P33 is REVERSED; graphs ON is correct everywhere and is a TTS-TTFB win.
- **`COSYVOICE_FIRST_HOP_ZH=5`** — REVERSED by a live A/B (P22); 0 is the baseline.
- **Avatar render "session degradation"** — did not reproduce once turns stopped overlapping: flat 0.44-0.52
  across 6 turns (P55). The CLAUDE.md note describes real user-observed behaviour, but the ONE repro anybody
  had was a driver artifact. If you chase it, get a repro that is not the harness first.
- **Barge-in latency** — measured: interrupted turns are identical to clean ones (P55).

## 6. Open, unclaimed

- **Barge-in TTS residue on the shared GPU (P56 — CONFIRMED live; clean fix ATTEMPTED + REVERTED).** A
  barge-in abandons `/tts/stream`; the server keeps generating the abandoned utterance's speech tokens, costing
  the NEXT turn **+0.9–1.1s** (verified with a production-faithful aiohttp repro; residue SCALES with the
  abandoned generation). The stop-flag + `vllm.abort_request` fix was implemented in the vendored `model.py`/
  `llm.py`, and REVERTED: the trigger worked, but CosyVoice drives vLLM's `step()` manually through one shared
  output queue, so a mid-batch abort broke the next turn (empty audio). **Do NOT re-attempt mid-batch abort as
  a quick patch** (full write-up: `docs/PROBLEMS-AND-FIXES.md` P56). Tractable angles instead: **cap reply
  length** (`OPENROUTER_MAX_TOKENS`, unread since the 2026-07-14 audit — bounds every residue AND fixes the ~50s
  ramble, pure-pipeline, no vendored risk), or a proper per-request lifecycle in the vLLM integration (a real
  effort). Severity is likely < the 0.9s worst case: production abandons ONE sentence, and the next turn only
  starts after the user speaks.
- **The intermittent ~1-in-7 +1.7s first-frame spike.** Not compute (GPU flat), not the flush window
  (identical on spiked vs clean turns), not the feed (first PCM on the wire in +0ms). Not yet caught with
  `[barge]` armed — it needs ~10+ turns to hit. Worth ~1.7s on ~13% of turns, vs TTS's 0.9s on 100%.
- **Replies run ~50s** for "what is AI". `OPENROUTER_MAX_TOKENS` was **never read** (2026-07-14 audit), so
  replies are uncapped. This is a product issue in its own right AND what makes the driver's default
  unworkable. Capping it would improve the felt product and the harness at once.
- **`--blead` vs the ICE handshake** (§4).
