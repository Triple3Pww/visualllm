# Handoff — barge-in TTS residue (confirmed live; fix attempted + reverted)

**Written 2026-07-16 (29th session).** This is the companion to `docs/TTS-FIRST-CHUNK-HANDOFF.md`
§6 and `docs/PROBLEMS-AND-FIXES.md` **P56**. It records, in full, the bug, the fix I tried, why I
reverted it, and where to go next — so nobody repeats the four dead ends.

## 1. The bug (CONFIRMED, live)

When you interrupt the bot mid-reply (a **barge-in**), the pipeline cancels the in-flight TTS
request — but the CosyVoice server does **not** stop. It keeps generating the abandoned utterance's
speech tokens on the shared GPU, so the **next** turn's TTS starts **~0.9–1.1s slower**.

Measured with `tts/cosyvoice-server/_bargein_residue_probe.py` (faithful: same aiohttp
`async with session.post(...)` cancel as `run_tts`, one persistent session, live server):

| next-turn TTFB (`早安`) after… | median | residue |
|---|---|---|
| a **drained** long request (clean) | ~0.46s | — |
| an **abandoned** long request | ~1.38s | **+0.9s** |
| an **abandoned SHORT** request | ~0.47s | **+0.0s** |

The short-abandon arm is the key control: **residue scales with the abandoned generation**, so it is
leftover speech-token work, not a fixed per-abandon cost, not an instrument artifact.

> **Reconciles with P55's "barge-in costs NOTHING":** P55 measured the *current* turn's first frame
> (+0ms — correct). This residue lands on the **next** turn's TTS, which P55 never measured. It also
> fits P55's still-open **~1-in-7 +1.7s** first-frame spike.

## 2. Root cause (in code)

- `run_tts` (`local_services/cosyvoice_tts.py`) streams inside `async with session.post(...)`. A
  pipecat `InterruptionFrame` cancels that generator → the `async with` exits → aiohttp closes the
  connection. That is the abandon.
- Server: `app.py::tts_stream(req)` never receives FastAPI's `Request`, so it cannot check
  `is_disconnected`. Its `StreamingResponse` stops being pulled, but…
- `CosyVoice/.../cli/model.py` runs the speech-token LLM on a **`threading.Thread` (`llm_job`)** that
  is not tied to the consumer. It keeps pulling `inference_wrapper` → `vllm.step()` to the end of the
  utterance, appending to a dict nobody reads. **That thread is the residue.**
- The vocoder half (`token2wav` in the main streaming loop) *does* stop on abandon; only the LLM
  generation leaks.

## 3. The fix I tried, and why it was REVERTED

**Plan:** on abandon, set a per-uuid stop flag; `llm_job` checks it, breaks, and calls
`self.vllm.abort_request` to drop the request from the engine. Implemented across the vendored
`model.py` (`tts_stop_dict` + `try/except GeneratorExit` around the streaming yields, cleanup moved
to `finally`) and `llm.py` (an `abort()` method).

**What each step taught me — all caught by instrumentation, not argument:**

1. **The trigger works.** On an aiohttp abandon, `GeneratorExit` fires and `llm_job` broke early
   (~350 tokens vs the full utterance). Logged and verified. **Residue did not move.**
2. **`abort_request` was a silent no-op.** Its signature in this build is
   `abort_request(request_ids: list[str])` (**vllm 0.23**). Passing a bare `uuid` string iterates it
   into single characters, matches no request, and raises nothing — so it looked like it ran.
3. **Fixing it to `abort_request([uuid])` broke the NEXT turn (empty audio).** CosyVoice drives
   vLLM's `step()` **manually**, routing **all** active requests' outputs through **one shared**
   `vllm_output_queue` (`llm.py::inference_wrapper`). Aborting a request mid-batch destabilises that
   shared loop (it also needs a `KeyError` guard at the router, because `step()` can emit a trailing
   output for an already-popped queue). The abandoned request is entangled with the next request's
   `step()`.
4. **Stopped (systematic-debugging Phase 4.5: 3+ fixes each revealing new coupling = wrong altitude)
   and reverted** both vendored files to pristine. Server restarted clean and re-verified: normal
   synthesis works, the next turn is intact, residue is back to its original ~0.93s. **Nothing left
   broken.** (Vendored files are gitignored, so the revert was manual; confirmed no remnants with
   `grep -E "P56|tts_stop|def abort"`.)

**Verdict:** a clean mid-batch abort is **not a small patch** — it is real vLLM-lifecycle surgery on a
manually-driven engine. Do **not** re-attempt it as a quick fix.

## 4. Severity (why reverting is defensible, not a cop-out)

In production, `run_tts` is called **per sentence** (LLM sentence-aggregation + `COSYVOICE_FIRST_PIECE`).
So a real barge-in abandons **one sentence's** remaining tokens, not the 90-char paragraph the probe
uses as a worst case. And the next turn only begins after the user speaks (VAD + STT + LLM), by which
time a single sentence's leftover has often already finished. **Felt cost is likely well under 0.9s.**

## 5. Where to go next (ranked)

1. **Cap reply length — the easy, safe win.** `OPENROUTER_MAX_TOKENS` has been **unread since the
   2026-07-14 dead-code audit**, so replies are uncapped (~50s rambles). Wiring it up **bounds every
   residue** (less to leak) **and** fixes the felt ramble — pure-pipeline (`pipeline/stages/llm.py`),
   **zero vendored risk**. This is the recommended next step.
2. **A proper per-request lifecycle in the vLLM integration** — the correct root fix, but a real
   effort: give `inference_wrapper` per-request abort that plays with the manual `step()` loop, or
   move off the single-shared-queue design. Only worth it if §5.1 proves insufficient by ear.
3. **Do NOT** re-try: passing `Request` into `tts_stream` alone (the stop still has to reach
   `llm_job`), or mid-batch `abort_request` as a quick patch (§3.3).

## 6. Artifacts

- **Probe (durable, in-repo):** `tts/cosyvoice-server/_bargein_residue_probe.py` — reproduces + the
  long-vs-short discriminator. Run against the live WSL server.
- **Full narrative:** `docs/PROBLEMS-AND-FIXES.md` **P56** (floor + residue + this attempt).
- **Context:** `docs/TTS-FIRST-CHUNK-HANDOFF.md` (the TTS row is ~70% fixed floor; this residue is the
  higher-value lead), `docs/LATENCY-MATRIX.md`.
- **Restart the WSL TTS server after any vendored edit** (a pipeline Restart does NOT reload it):
  `bash /mnt/e/Claude/VisualLLm/tts/cosyvoice-server/run_vllm_server.sh` in WSL; ~20–25s to `/health`.
  Gotcha hit this session: a background job started via `wsl -e bash -c "... &"` dies when the
  transient WSL session tears down — launch it as a persistent foreground process instead.
