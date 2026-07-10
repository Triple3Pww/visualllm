# CosyVoice 3 vs CosyVoice 2 — A/B design (2026-07-10)

## Question

Is `Fun-CosyVoice3-0.5B-2512` affordable enough, on this shared 16GB card under live
MuseTalk render, to replace `CosyVoice2-0.5B` as the pipeline's TTS baseline?

Quality is not in doubt (the paper reports 44–51% relative content-consistency gains).
The open question is **cost**, and it lands in the worst possible place for us: the LM
half is the same 0.5B and is already vLLM-accelerated, while the flow-matching decoder
grows 100M → 300M and changes from chunk-aware causal CFM to a DiT. The decoder is the
part vLLM does **not** accelerate and the part that contends with MuseTalk.

## Verdict rule

Two phases, with a hard gate between them.

1. **Latency gate (probe decides).** v3 **fails** if first-chunk TTFB median regresses
   more than ~0.3s over v2's ~1.94s baseline, or if max exceeds ~4.0s. Either would push
   zh TTFO back over the 3s target once the `MUSETALK_LEAD_FRAMES=14` cushion fill is added.
2. **Eye (user decides).** Only if the gate passes: render matched zh clips through the
   delivered path; the user watches and makes the call.

This ordering is deliberate. P19/P22/P33 all ended with the probe passing what the eye
rejected — so the probe is allowed to **kill**, never to **approve**.

## What already exists (verified 2026-07-10)

- The vendored `CosyVoice/` is at the `cosy3_pr` merge (`ace7c47`). It has `CosyVoice3`,
  `CosyVoice3LM`, `CosyVoice3Model`, and `AutoModel` dispatching on `cosyvoice3.yaml`.
  **No repo upgrade needed.**
- `CosyVoice3Model` subclasses `CosyVoice2Model` → inherits `load_vllm` → the P18
  `RasLogitsProcessor` registration is on v3's path too. `CosyVoice3LM` subclasses
  `Qwen2LM`, the arch registered as `CosyVoice2ForCausalLM`.
- `COSYVOICE_FLOW_TRT` defaults to `0`, so v2's flow decoder already runs plain PyTorch.
  **No TRT confound** — both arms are PyTorch flow, apples to apples.
- Missing: only the v3 weights. The `vllm/` subdir is generated at load by
  `export_cosyvoice2_vllm`.

## Design

### Components

| Piece | Change |
|---|---|
| `tts_engine.py` | Swap hardcoded `CosyVoice2(...)` → `AutoModel(...)`. Drop the `load_jit` kwarg (`CosyVoice3.__init__` has no such parameter; it is `False` today). `COSYVOICE_MODEL_DIR` **already exists** at `tts_engine.py:67` — no new env var needed. |
| weights | `Fun-CosyVoice3-0.5B-2512` → `CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512/` |
| contention rig | MuseTalk server (`musetalk` env, production env vars) + `scripts/_drive_frames.py output/reply_concise.wav 12` looping. Real renderer, deterministic workload. Pipeline/WebRTC not needed for the gate. |
| probes | `_ttfb_variance.py` (8 openers × 4 rounds = 32 samples/arm/cycle); `_zh_audio_ab.py` extended with a **leading**-silence metric (free P34 breath check). |
| driver | Cycles the WSL server between model dirs, waits `/health`, runs both probes, appends `output/cv3_ab.json`. |

Baseline behavior is unchanged when `COSYVOICE_MODEL_DIR` is unset. v3 never touches the
live `.env`.

### Data flow

`probe → :8001 /tts/stream` (first-byte = TTFB) and `→ :8001 /tts` (full wav → audio stats).
Same HTTP contract both arms; same WSL IP rule (never `localhost`).

### Sequencing

Arms are sequential (see *Rejected alternatives*), so thermal drift and background load are
confounds. Mitigation: run **A,B,A,B** and compare medians *within* each cycle, not across
the whole run.

### The null test, first

Before any comparison, run the harness as **A,A** — v2 against itself. **Pass = the two v2
arms' medians differ by ≤ 0.15s** (half the gate threshold). If they differ by more, the
harness is measuring drift rather than models and every subsequent number is noise: stop and
fix the rig. Costs one extra cycle. This is the check that would have caught the P32 error early.

### Controls

- **Same reference clip both arms** (`pro_ref.wav`). P18: the reference clip drove the zh
  choppiness we had blamed on the model. Vary it here and we A/B the voice, not the architecture.
- **Same `gpu_util`, same `COSYVOICE_VLLM_EAGER=1`, same first-piece/hop settings** both arms.
  If v3's larger DiT forces a `gpu_util` change, the change applies to **both** arms or the
  run is void.
- **Load order**: CosyVoice (vLLM) before MuseTalk, every cycle (P15).

## Risks — reported as blockers, not as results

1. ~~**`<|endofprompt|>` assertion.**~~ **RESOLVED, no code change.** `CosyVoice3LM` asserts token
   `151646` is in `prompt_text` (`cosyvoice/llm/llm.py:591`), but `<|endofprompt|>` is already in the
   tokenizer's `allowed_special` (`cosyvoice/tokenizer/tokenizer.py:249`) and `prompt_text` comes from
   the `COSYVOICE_PROMPT_TEXT` env (`tts_engine.py:48`). The v3 arm sets that env with the token appended.
2. **vLLM export may reject v3's `llm.pt`.** If it does, the fallback is v3 on PyTorch — and
   **that number is not reportable as "v3 is slow."** It is a blocked test, and must be
   reported as such, not as a rigged comparison.
3. **RAS must be confirmed to fire under v3**, from the log — not merely confirmed to register.
4. **VRAM.** v3's DiT is 3× the CFM parameters. If load crashes with "No available memory for
   the cache blocks," that is a *finding about affordability*, not a harness bug.

## Rejected alternatives

- **Two servers, interleaved request-by-request.** Statistically cleanest; kills drift
  entirely. Does not fit: vLLM at `gpu_util 0.3` is ~4.9GB, MuseTalk ~5GB, plus v3's larger
  flow. Fitting two engines requires dropping `gpu_util` to ~0.12, which changes the variable
  under test. **The GPU budget *is* the experiment.**
- **In-process, no server, no vLLM.** Bypasses the thing that makes v2 fast. Kept only as a
  fallback if the vLLM export path chokes.

## Out of scope (YAGNI)

- Auto-scoring the polyphone sentences (v3's pronunciation-inpainting claim). Judging
  pronunciation needs an ear or an ASR; wiring sherpa in is scope creep. They will be
  synthesized and included in the phase-2 clips for the user's ear.
- Rebuilding a TRT engine for v3's DiT. Both arms stay PyTorch, matching the live `.env`.
- Any change to the live `.env` baseline or to `pipeline/metrics.py`.

## Phase 2 (only if the gate passes)

Render matched zh clips through the **delivered** path (`MUSETALK_DUMP_DELIVERED`), never an
offline render fed a repaired PCM copy — P40's rule that the reference must not share the
suspect input. Same zh text, same reference voice, both arms. Hand the user two files to watch.

## Success criteria

- The A,A null test shows no significant delta. (If it does: stop, fix the harness.)
- Both arms confirmed running vLLM + RAS, same `gpu_util`, same reference clip.
- A gate verdict backed by ≥32 TTFB samples per arm per cycle, reported as median/max/stddev.
- Either a clear FAIL with numbers, or two clips on disk for the user.
