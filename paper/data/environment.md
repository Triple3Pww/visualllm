# Measurement environment freeze

_Campaign date: 2026-07-16 (zh + en TTFO campaigns, Tasks 2–3 of the paper plan)._

## Code state

- Branch: `chore/cleanup-and-tts-merge`
- HEAD at campaign time: `cdb66ec2493332db36a24e7cfb7a16bdf5e79fa3`
- Uncommitted working-tree changes present (recorded honestly — these files were live during
  measurement): `.env.example`, `CLAUDE.md`, `STATUS.md`, `WORKFLOW.md`,
  `docs/PROBLEMS-AND-FIXES.md`, `local_services/cosyvoice_tts.py`,
  `local_services/musetalk_video.py`, `local_services/sherpa_stt.py`, `pipeline/config.py`,
  `pipeline/main.py`, `scripts/launch.ps1`, untracked `archive/_tts_stop_order_test.py`.
  (These are the session-25/26/27 fixes — P51 proto-2 client, P52 producer-side PCM carry,
  P53 BotStopped ordering — i.e. the measured system IS the current verified baseline.)

## Hardware / OS

- GPU: NVIDIA GeForce RTX 5060 Ti, 16311 MiB, driver 591.44 (Blackwell)
- Windows 11 Pro 10.0.26200 + WSL2 Ubuntu (CosyVoice TTS on vLLM inside WSL)

## Live `.env` configuration (the knobs that shape the numbers)

| key | value |
|---|---|
| LANGUAGE | zh (switched to `en` only for the en campaign, then restored) |
| TTS_PROVIDER | cosyvoice |
| COSYVOICE_MODEL | v2 (CosyVoice2-0.5B on vLLM/WSL, CUDA graphs ON) |
| OPENROUTER_MODEL | meta-llama/llama-4-scout |
| OPENROUTER_PROVIDER_ONLY | Groq |
| MUSETALK_SYNC_MODE | steady (video-master) |
| MUSETALK_FPS | 12 |
| MUSETALK_SIZE | 512 |
| MUSETALK_TRT | 1 |
| MUSETALK_LEAD_FRAMES | 14 |
| COSYVOICE_FIRST_PIECE | 1 |
| COSYVOICE_FIRST_PIECE_ZH | 1 |
| FILLER_WORDS | 0 |
| MUSETALK_SPLIT | 1 |

## Protocol notes

- 10 runs per language, each a fresh probe connection (= fresh session; the single-client avatar
  server drops the previous session), `python -m scripts.measure`.
- zh mic wav: `output/_zh_q_def.wav` (`q_ai.wav` stopped transcribing on Deepgram 2026-07-15).
- en mic wav: `output/_en_q.wav` — transcribes as "HELLO CAN YOU HEAR ME WHAT IS THE WEATHER
  LIKE TODAY" (verified before the campaign, 2026-07-16 00:16, TTFO 2.02 s on the check run).
- After the en campaign `.env` `LANGUAGE` was restored to `zh` and verified with a throwaway
  zh run (2026-07-16 00:27, question transcribed in Chinese, TTFO 3.84 s).
- Known outlier source: Groq provider congestion adds ~2–3 s to the LLM row on occasional runs;
  such runs are kept (p95 is meant to see them) and noted below.

## Outliers / anomalies observed

- **zh campaign (2026-07-16 00:04–00:13):** TTFO per-run 2.28, 2.30, 2.27, 2.53, 2.14, 3.94,
  3.31, 4.94, 3.75, 3.53 s. Runs 1–5 cluster at 2.1–2.5 s, runs 6–10 at 3.3–4.9 s; the stage
  deltas attribute the entire slowdown to the steady lead-hold (0.60–0.64 s early vs
  0.65–2.39 s late) — render-side shared-GPU variance, not the cloud (TTS first-chunk stayed
  0.81–1.04 s across all 10).
- **LLM row caveat (applies to ALL runs in this campaign):** the sentence-1 flush delta is
  ~0.01–0.03 s in every run because the synthetic mic wav VAD-splits — the LLM starts on the
  first VAD segment and generates during the tail of the recorded utterance, so its cost
  overlaps user speech instead of following t0 (the documented harness caveat: "a real human
  turn populates it"). The campaign's LLM row therefore measures overlap, not the LLM hop; the
  citable real-turn LLM numbers are the development-time Groq-pin measurements (zh 0.80 s / en
  0.67 s median). The paper must label this.
- **en campaign (2026-07-16 00:17–00:25):** TTFO per-run 2.30, 2.26, 2.78, 1.88, 2.83, 2.24,
  2.00, 2.16, 1.86, 2.12 s — tight band, no outliers; the zh campaign's late-run lead-hold
  inflation did not recur (different question length and a different GPU-load moment; treat the
  zh p95 as the honest tail, not a fluke to discard).
