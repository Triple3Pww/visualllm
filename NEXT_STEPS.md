# What To Do Next — VisualLLm

Plain checklist for the next session. Do these in order.
(Full context is in `STATUS.md`. The system already works — this is what's left.)

> **NEW (2026-06-10):** the avatar now runs **locally** (Phase 3, MuseTalk on the
> 5060 Ti) instead of Simli cloud. It needs **two processes** and a one-time
> in-browser visual check. See step 0 and step 1.

---

## 0. Start the system (now TWO processes for the local avatar)
- Open the project: `cd E:\Claude\VisualLLm`
- **Start the avatar server FIRST** (its own conda env):
  `local_services\musetalk_server\run_server.bat`
  — wait for `Uvicorn running on http://0.0.0.0:8002`.
- **Then start the pipeline:** `python -m pipeline.main`
- In the RDP desktop browser, open `http://localhost:7860/client`
- **Hard-refresh: Ctrl+Shift+R** (clears the old cached page)
- Wait for the avatar's face to appear, then talk.
- (To use the old Simli cloud avatar instead: set `AVATAR_PROVIDER=simli` in
  `.env` — then you don't need the server.)

## 1. Check the loading screen works
- When you open `/client`, you should see **"Loading the avatar…"** with a spinner.
- It should stay until the avatar's face appears, then disappear.
- ✅ If yes → done with this.
- ❌ If it never disappears, or appears wrong → tell Claude; fix is in the bottom
  of `pipeline/main.py` (the overlay script).

## 2. Rotate the OpenRouter key (security)
- The key was briefly saved in a shared file.
- Go to openrouter.ai → Keys → delete the old key → create a new one.
- Put the new key in `.env` (the line `OPENROUTER_API_KEY=...`). Do NOT put it
  in `.env.example`.

## 3. Switch to Mandarin (zh-TW) — the real goal
This is your research target. It's mostly a `.env` change:
- Set `LANGUAGE=zh`
- Set the LLM to a strong Chinese model, e.g.
  `OPENROUTER_MODEL=qwen/qwen-2.5-7b-instruct` (or a deepseek model)
- Set `ELEVENLABS_VOICE_ID` to a voice that sounds good in Mandarin
- (Optional, better for Thailand latency + zh quality) switch STT/TTS to **Azure**:
  `STT_PROVIDER=azure`, `TTS_PROVIDER=azure`, add `AZURE_SPEECH_KEY`,
  `AZURE_SPEECH_REGION=eastasia`
- Run `python -m scripts.preflight` to confirm keys load, then test.

## 4. Local MuseTalk avatar — ✅ DONE, now just check it
The avatar runs locally now (no more Simli cloud lag). The implementation,
weights, conda env, and `.env` wiring are all done. What's left for you:
- **Visual check (most important):** open `/client`, speak, and watch the lips.
  - Do they move in time with the words? Judge **audio/video sync** — if the lips
    clearly trail the voice (~0.5–0.8 s), tell Claude: the fix is to delay the
    audio in `local_services/musetalk_video.py` so it's emitted with each video
    frame (see the caveat in `PLAN.md`).
- **Use your own face:** replace `assets/avatar.png` with a clear front-facing
  portrait, then delete `local_services\musetalk_server\avatar_cache\` so it
  re-prepares from the new photo. Restart the server.
- If the avatar server won't start, the pipeline logs a clear "MuseTalk server
  not reachable" warning telling you to start it first.

## 5. (Optional) Push latency even lower
- Try Gemini through Google's own API (Asia region) instead of OpenRouter (US),
  to shorten the network hop. Needs a free Google AI Studio key.

---

### Quick reference
- Start: `python -m pipeline.main` → open `/client`
- Check setup: `python -m scripts.preflight`
- Measure speed: `python -m scripts.bench_latency --stage llm`
- All settings live in `.env`. Full status in `STATUS.md`.
