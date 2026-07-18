# Handoff — doc-accuracy cleanup: retire what a later root-cause fix superseded

**Written 2026-07-16 (30th session).** Companion to `docs/PROBLEMS-AND-FIXES.md`. The user is
continuing this next session. Read §1 (the request), §2 (the rule and its LIMIT — this is the part
that matters), then §5 (what's left).

## 1. The user's request (verbatim intent)

> *"I will read the problem and fix and give you the thing that I think it might not useful anymore
> because some core problem is already fix the root cause so I want you to test by run the in before
> and after remove that part."*

and, later, the generalised rule:

> *"if the root cause fix by future then remove it"* … *"now delete everything that superseded by
> future fix"*

So: the user reads `PROBLEMS-AND-FIXES.md`, nominates a part they suspect is dead (because a LATER
fix closed its root cause), and **the assistant must verify by measurement — before/after — not by
argument**, then remove it if genuinely dead. **He hands over the candidate; you do the proving.**

## 2. THE RULE, AND THE LIMIT THAT THE SWEEP FOUND (read this before deleting anything)

The rule works — it retired P7/P8/P10. But a full sweep (30th session) found **it stops working after
those**, because of a distinction that is easy to miss:

> **What later fixes usually superseded was the MECHANISM or the DETECTION — not the FIX.**
> The fix is often still running. Deleting its section removes the *why* from code that still executes.

Concretely, the sections that **look** superseded but are **not** (all verified in code):

| Section | Looks dead because | Why it is NOT dead — the receipt |
|---|---|---|
| **P3** (steady screech) | P52 fixed odd-bytes at the producer; `_align_even` deleted | `pipeline/main.py:127` → `_relax_bot_vad_stop_timeout()  # steady-mode screech fix` is **LIVE**. Only the `_align_even` half went |
| **P39** (held-frame dups) | P40 superseded its *explanation*; P51 deleted its byte-compare | `musetalk_video.py:455-462` **still drops held frames** and cites P39 for *why*. P51 changed only the DETECTION (kind flag vs byte-compare) |
| **P33** (CUDA graphs / zh) | Header is struck through, "REVERSED" | Not superseded **by a fix** — its *verdict* was reversed by the live eye. `CLAUDE.md:77` cites it for a live rule |

**The second half of the limit:** the superseders are *named after* their subjects —
P52 = *"the odd-byte class (**P3** screech, P34 crash, **P40** noise) fixed at its ONE source"*,
P53 = *"BotStopped never fired under steady sync (**the P11 root cause**)"*. They are **closure
records, not standalone explanations**. Delete the subject and the closure is a sentence without a
noun. The chain *bug → wrong theory → real root cause → closure* **is** the institutional memory —
and per the paper verdict (`project_visualllm_paper_publishability_verdict` memory) that chain is the
paper's real contribution: `paper/draft.md:410` already draws §7's lesson from **P40**.

**Litmus test before deleting a section — all three must be true:**
1. `grep` the codebase: **is the fix still running?** (If yes → keep. Move the *why* into the code instead.)
2. **Does anything cite it** — code comments, `CLAUDE.md`, `WORKFLOW.md`, `STATUS.md`, `paper/`, another P-section's *title*?
3. **Is its live knowledge duplicated elsewhere**, so nothing is lost?

P7/P8/P10 passed all three. P3/P33/P39 fail #1. P11/P40 fail #2.

## 3. DONE this session (all docs-only except one docstring; nothing committed)

**Verified LOAD-BEARING — do not remove (measured, not argued):**
- **P1 `cudnn.benchmark = False`** — ran the 2×2 (`MUSETALK_PROFILE=1` + `_drive_frames`, 13.56 s reply):
  `TRT=0,bench=True` = **16,131 ms first segment, twice in one reply, 88/163 frames, −6.1 s drift**;
  every other cell clean. **Per-turn, not first-turn; steady cannot mask it** (it would pause the voice
  ~6 s); **the catastrophe is confined to the `TRT=0` fallback** — under the live `TRT=1` the flag only
  buys ~0.5 s of whisper re-tune. Trap: the flag **defaults to `False`**, so deleting the line looks
  safe while deleting only the guard. Matrix now recorded in **P1**.
- **P9 `samples_for_frames`** — `MUSETALK_FPS=12` does **not** divide 16000, so the ceil sizing is what
  makes fps=12 legal. Repro re-run: OLD = 7 frames/seg, NEW = 8; e2e old=142 vs new=162. Live drive at
  baseline: **162/163**. Note recorded in **P9**.

**REMOVED (all three passed the litmus test — they described code that no longer exists):**
- **P7** (locked/steady froze the voice → "live remains the default") — contradicted the shipped
  `steady` baseline. Its "never re-lock the voice" rule survives in CLAUDE.md/STATUS/WORKFLOW/
  workflow-diagram (which cite STATUS, not P7). Dangling `(P7)` tags dropped from `PRESENTATION.md/.html`.
- **P8** (TTS dead = ElevenLabs out of credits) — the `elevenlabs`/`deepgram` branches were removed
  2026-07-14; zero inbound refs; its symptom now misdirects (today "won't talk" = CosyVoice/vLLM down, P15).
- **P10** (leftover-audio blip) — root cause **removed by P51** (`f2baf54`): proto 2 pairs voice to each
  frame's `audio_pos`, so there is no cursor cap to floor. All **17 refs** rewired across 6 files;
  **4 were actively false** (claimed the deleted cap was "kept"/live — incl. **`.env:131`**); six
  `P9/P10` pairs actually described P9's ceil sizing → corrected to `P9`.
  **The one thing preserved:** the code cited P10 for *why there is no cap*, so that warning now lives
  **in the code itself** (`musetalk_video.py::_advance` docstring: *"Do NOT add one back: a floor()'d cap
  strands the turn's final audio sub-frame…"*). This is the pattern to reuse: **relocate the rule into
  the code it governs, then delete the section.**

**CORRECTED (kept, but the claim was wrong):**
- **P6** — retitled to what it actually fixes: **TTS TTFB 3.4s → ~1.1s**, *not* lip-lag. In the current
  `steady` baseline the **sync mode owns lip-lag** (voice paced to rendered frames ⇒ ~0 by construction),
  so a faster TTS is a TTFB/TTFO win only. Its old Symptom ("lips trailed 1.5–2 s") was the **then-default
  `live`/audio-master** mode's symptom — the user caught that it read as a steady symptom; Symptom now
  states the mode-independent cost (~3 s before first sound). Synced in `CLAUDE.md` + `STATUS.md`.

**Files touched:** `docs/PROBLEMS-AND-FIXES.md`, `CLAUDE.md`, `STATUS.md`, `WORKFLOW.md`,
`docs/PRESENTATION.md`, `docs/PRESENTATION.html`, `.env` (one stale comment),
`local_services/musetalk_video.py` (**docstring only** — verified comment-only, file parses).
Sections now run **P6 → P9 → P11**; numbering deliberately **not** renumbered (P11–P56 are cited everywhere).

## 4. Method notes (what made the verification honest)

- **The 2×2 beats the A/B.** P1 only became clear when TRT was a second axis — the flag's value is
  invisible on the live path and catastrophic on the fallback. Ask "which code path does this actually
  guard?" before concluding "no effect = dead."
- **A default can fake a passing test.** `cudnn.benchmark` defaults to `False`: removing the line and
  measuring "no change" would have *proved nothing*. Toggle to the dangerous value instead.
- **Docs-vs-code is the real bug surface here.** Every real finding this session was a doc asserting
  something the code contradicted (`audio_cap` "is live"; WORKFLOW telling you to *re-apply* a cap proto 2
  deliberately deleted). `grep` the claim against the code before trusting any section.
- Test harness used: `C:\Users\MARU\AppData\Local\Temp\claude\…\scratchpad\p1_run.ps1` (scratchpad, may be
  gone). It stops :8002, restarts the server with prod env + `MUSETALK_PROFILE=1` + a TRT override, drives
  `_drive_frames`, prints `[profile]`. **Restore the stack afterwards** — it leaves :8002 down.

## 4c. DONE 2026-07-18 (32nd session) — **P16, and the rule's FOURTH failure mode: a fix nominated dead that is only MASKED**

The user nominated P16 ("progressive drift is fixed by a future fix, so the section is dead"). **Wrong
diagnosis, right instinct.** No later fix closed P16's root cause — the drift is **masked by margin**, and
the fix (`MUSETALK_TRT=1`) is what supplies the margin. **KEPT, corrected, not deleted.**

**The 2×2 (`MUSETALK_TRT` × contention) at the LIVE config (SIZE=512/SPLIT=1/fps=12), `_drive_frames`,
P16's own 2.88/5.69/13.56 s series:** all four cells identical, **+0.357 s flat**, with TRT=0 or 1, under a
100% hog OR real CosyVoice generating. A naive A/B stops here and deletes P16. **That is the P1 trap** — a
config whose value is invisible on the live path. Profiler shows why: TRT=0 renders ~455 ms/seg vs TRT=1's
~171 ms, but at fps=12 **both** clear the 667 ms budget (PyTorch by just 4%), so drift is identical. The
difference is pure headroom, unseen until spent.

**THE FOURTH FAILURE MODE — the positive control had to be CONSTRUCTED, and the obvious one was broken.**
- `_gpu_contention_hog.py` claims "N=4096 forces render < 12fps". **It does not, and N=8192 steals no more
  than N=4096** (MuseTalk gpu 567 vs 569 ms — identical). Windows **WDDM** time-slices the two processes to
  a ~50% floor regardless of the other's weight, so the hog **cannot produce the failure it was written for**
  — it is a control that can only pass. Real CosyVoice (the actual thing) also failed to starve it. This is
  P15's lesson from the other side: *a pass proves nothing unless the rig can still FAIL* — and here neither
  contention tool could make it fail.
- The working positive control tightens the **budget**, not the load: drive at **fps=25** (budget 320 ms/seg,
  under PyTorch's 455). Then P16 returns in full, same code, same hour:
  **TRT=0 → +0.64 / +1.95 / +4.04 s (101 held frames); TRT=1 → +0.36 / +0.35 / +0.32 s flat (8 held).** That
  reproduces the 2026-07-01 signature (+0.37/+1.35/+3.94) almost exactly and pins causality on TRT.
- **The honest verdict:** at today's fps=12, TRT buys **margin** (contended headroom 1.04× → 2.05×), not a
  visible drift fix — and the 1.04× PyTorch cell is **24 ms/seg** from collapse, measured with an idle CPU.
  Higher fps, bigger SIZE/BASE_MAX, or a weaker card puts P16 back. `docs/GPU-REQUIREMENTS.md` already relies
  on this ("PyTorch fallback too slow"), so the section is live-cited → **litmus #1 fails, KEEP**.

**Also corrected: the SYMPTOM was `live`-mode language.** "Lips fall progressively behind the voice" cannot
happen under the `steady` baseline — there the same underflow is a **voice pause** (`musetalk_video.py:459`:
*"holds the last real frame and the voice pauses IN SYNC … instead of drifting"*). Same fix as P6's symptom
correction (30th session). Do not conclude "no drift = no problem" under steady; look for the hold.

**Files touched (docs + 2 comment-only script edits; both `.py` re-parsed, ASCII-clean):** `docs/PROBLEMS-AND-FIXES.md`
(P16 Symptom + a RE-MEASURED block + a forward-pointer over the 2026-07-01 prose), `.env:134` (the live comment
claimed "+3.9s under contention" — false at fps=12; now states margin + the fps=25 control), `CLAUDE.md`,
`WORKFLOW.md` (troubleshooting row **and** the knob table), `docs/GPU-REQUIREMENTS.md`, `docs/PRESENTATION.md`
+ `.html`, `paper/draft.md` (§5.3 asserted TRT was "the difference between drifting seconds behind and holding
≥12 fps" — present-tense **and** self-contradicting §5.2's own "pauses rather than drifts"; rewritten to the
margin framing + turned the near-miss into §7's ablation-at-the-operating-point point; ablation table row
updated), `scripts/_gpu_contention_hog.py` + `scripts/_drive_frames.py` (docstrings — the hog's "forces render
< 12fps" was a landmine that would mislead the next session's contention test). STATUS.md's 2026-07-01 session
log was **marked** superseded at the block top, history left intact (same policy as 4b). **Stack was left down
mid-session by a stray `Restore CosyVoice…` background task + the test's :8002 cycling; fully restored
(WSL TTS → avatar → pipeline → config panel, all four health-checked green).**

## 4b. DONE 2026-07-17 (31st session) — **P15, and the rule's THIRD failure mode**

**P15 was TWO documents stapled together, and they died differently.** The user nominated "P15"; CLAUDE.md
cites P15 for the *load order*, while the section's TITLE is about *zh latency*. Check which half a citation
means before you touch it — the 20 refs split across both.

- **§1 zh first-chunk penalty — REFUTED by measurement.** Claim: zh TTFB 2.0–2.75s vs en 1.0–1.5s. Re-ran
  `_ttfb_variance.py --rounds 3` (n=24) at the live baseline: **zh ≤1.79s, en up to 2.25s — the slowest
  cases are ENGLISH**, distributions fully interleave. **Length predicts TTFB, not language** (P56's law).
  Killed by P23 (zh comma-split) + P27 (graphs). **P30 had already written the refutation at line 1249 —
  in this same file — and nobody propagated it**, so README/SETUP/WORKFLOW/STATUS kept shipping the dead
  claim for 12 days. Its hop=5 verdict survives but **P22 owns it**; its `COSYVOICE_PACE_RATE` sentence was
  independently false (that knob is OFF).
- **§2 load order — NO LONGER REQUIRED, measured.** vLLM started **second** onto a MuseTalk-occupied card:
  loads at `util=0.07` **and at `0.30`** (the config that originally crashed). vLLM 0.23.0 budgets
  `total_card × util` and **does not charge other processes**; gate is `total × util ≤ free_at_startup`, so
  the wall is **util ≈ 0.79**. "300–400 MB free" was really **9538 MB**.

**THE NEW METHOD LESSON — the negative control is not enough; you need a POSITIVE one.**
The §2 test *passed* (loaded second). But a pass proves nothing unless the rig can still FAIL. The original
crash **could not be reproduced** (util=0.3 loads fine now), so the passes were suspect until `util=0.9`
produced a real `ValueError: Free memory … less than desired` — **that** is what makes "it loads" evidence
instead of a hopeful green. Sibling of P1's "a default can fake a passing test" (§4): there, toggle to the
dangerous value; here, **construct a condition that must fail, and check it does.**

**Corollary — "I fixed it" ≠ "I know what fixed it."** Do NOT credit the 2026-07-15 util drop 0.3→0.07 for
killing the load-order rule: at 0.3 it loads fine *today*, so the util was never the deciding variable
(most likely the vLLM upgrade changed the accounting). The honest claim is **"does not reproduce at any
util you'd use"** — not a causal story you didn't test.

**Scope discipline that mattered:** only the TTS was re-measured, so §1 refutes the **attribution**, not the
paper's e2e zh 2.92 / en 2.20 gap. The section now says so explicitly. Don't let a component measurement
retire a system-level claim.

**Files touched (docs + 2 comment-only code edits; `server.py` re-parsed, still ASCII-safe):**
`docs/PROBLEMS-AND-FIXES.md` (P15 rewritten, kept — see below), `CLAUDE.md` (load-order banner + the
"Chinese first-chunk is slower" heading, which **contradicted its own next line**), `README.md` (the
**public** one shipped the dead claim), `SETUP.md` ×2, `WORKFLOW.md` ×2, `STATUS.md` (dated session logs
**marked** superseded, not rewritten — history stays), `local_services/README.md`, `docs/PRESENTATION.md`
+ `.html`, `paper/draft.md:179` (**the paper asserted "load order matters"** — now "defensive rather than
required"), `local_services/config_panel/server.py` (docstring + a **user-facing failure message that told
you to chase load order when TTS is down** — the exact debugging cost of a stale claim).

**P15 was KEPT, not deleted** — it fails litmus #1: `launch.ps1` + `config_panel::restart_avatar` still run
the ordered restart, so the section is the *why* for live code. It is now titled as RESOLVED, carries both
2×2 tables, and states the util ≈0.75 threshold at which the order becomes required again.

## 4d. DONE 2026-07-18 (33rd session) — **the full sweep of §5: P11/P13/P40 RETIRED to an ARCHIVE file, `.env` corrected**

The user chose **full sweep + clean removal (no stubs) + direct surgery**. All of §5's open items closed:

- **New file `docs/PROBLEMS-AND-FIXES-ARCHIVE.md`.** Sections whose **fix AND the code it described are both
  gone** now live there (policy header explains why each was retired). P-numbers are **stable** in the
  archive so a `(P11)`/`(P40)` ref anywhere in the repo still resolves — that is why NONE of the ~90 refs
  across 16 files were touched (they were never going to dangle). Main-doc header gained a one-line pointer.
- **P13 (MOSS)** — pure P8-pattern: `TTS_PROVIDER=moss` + `local_services/moss_server/` were removed
  2026-07-14 (`tts.py` now RAISES on unknown provider; dir gone). Proven dead by **grep, not measurement** —
  the code doesn't exist, so there is nothing to measure (same basis as P8). STATUS.md's 2026-06-29 MOSS
  session log **marked** superseded at its top, history left intact (§4b policy).
- **P40 (odd-byte generic-mouth)** — followed the handoff's own recommendation: **merged the live mechanism
  INTO P52 FIRST**, then moved the full section. P52 previously explained where odd bytes come *from*; it now
  also carries what a dropped byte *DID* (Whisper-of-noise → generic flap, voice stays clean = the trap) +
  the paper's §7 metrology lesson (a reference sharing the suspect input can never fail), pointing to the
  archived P40 for the full evidence tables. So litmus #3 (knowledge duplicated elsewhere) is now satisfied —
  it wasn't before the merge, which is exactly why the handoff said merge-first.
- **P11 (echo-guard stuck-mute)** — moved as-is; its live knobs already live in CLAUDE.md and P53's banner,
  and P53's title "(the P11 root cause)" resolves to the archived section.
- **`.env:126–130`** — user **approved** the rewrite. The `live`-mode NOTE ("*lips trail ~1.5-2s … Accepted:
  immediate voice, lips best-effort*") under `MUSETALK_SYNC_MODE=steady` is replaced: keeps the measured
  render-vs-CosyVoice floor (0.77s render vs 1.2s-over-1.5-3s delivery), states the steady consequence
  (synced-start **delay** / brief voice pause, NOT a trail), notes the live face in one clause. Same
  symptom-mode correction as P6/P16.

**Method note for this one:** P13 is the reminder that the rule has a *fast* path — when the CODE is gone
(not just superseded), grep is sufficient proof and no before/after harness run is needed. The measurement
rule is for fixes that still RUN (P1/P9/P15/P16); a removed provider is the P7/P8/P10 lane.

**Files touched:** `docs/PROBLEMS-AND-FIXES.md` (3 sections removed, header pointer, P52 merge),
`docs/PROBLEMS-AND-FIXES-ARCHIVE.md` (new), `STATUS.md` (MOSS log marked), `.env` (NOTE rewrite).
Seams verified clean (P9→P12→P14, P39→P41; single `---` between neighbours). **Nothing committed.**

## 4e. DONE 2026-07-18 (33rd session, same day) — **full-doc staleness audit: every kept section re-checked against code**

After the §4d removals, the user asked to "read every problem and make sure it's not stale or superseded." Read
all 50 live sections; grep-verified each falsifiable claim against the tree (not argued). Most were current or
already self-corrected by their own reversal banners (P33/P34/P39/P42 — the good pattern). **7 stale claims
found + fixed**, each with a code receipt:

1. **P18** — files pointed at `E:\Claude\cosyvoice-local-tts` (retired) + "Not yet git-committed". The RAS fix
   is live at `tts/cosyvoice-server/` (subtree-merged, P49). → repointed, dropped the stale status.
2. **P32 / P33** — probe paths `cosyvoice-local-tts/_ttfb_variance.py` / `_zh_audio_ab.py`. Both live tools now
   under `tts/cosyvoice-server/`. → repointed.
3. **P51:2158** — listed `_align_even`/`_srv_carry` as "load-bearing and UNTOUCHED", but **P52 deleted both**
   two sessions later (musetalk_video.py comments confirm "GONE"). → removed from the list.
4. **P20** — `COSYVOICE_PACE_RATE` "currently 1.3" (code default is `0`/off, app.py:120) + two "untried" levers
   that **P25 built and rejected**. → fixed the value, added the P25 forward-pointer.
5. **P21** — "`OPENROUTER_PROVIDER_ONLY` is NOT in the config-panel" — it **is** now a curated "Pin LLM backend"
   field (server.py:63). → corrected.
6. **P37** — dangling "the proper cure is a TTS-frame mic mute (P11's future option)"; that need was closed by
   **P53**. → pointed at P53/ECHO_GUARD.

**Left as history (git owns it):** dated commit hashes pushed to the old repo (P42/P43), P28's env-snapshot
filename, P36's "unpin OPENROUTER_PROVIDER_ONLY" 429-fix (later re-pinned; framed as 11th-session history under
a removal banner). **Method reminder this reinforced:** the stale claims were almost all *pointers to live code*
(a moved path, a removed guard, a changed default) — grep the pointer, don't trust the prose. Files touched:
`docs/PROBLEMS-AND-FIXES.md` only.

## 4f. REVERSED 2026-07-18 (same day) — the §4d ARCHIVING was un-done; the approach changed to a separate clean file

The user **created `docs/PROBLEMS-AND-FIXES-CLEAN.md`** — a current-only "why the code is this way" map,
organised by subsystem, that states each truth once at its final resolution (reversals/dead-ends dropped).
That is the clean read now. So **`PROBLEMS-AND-FIXES.md` reverts to being the full, messy, lossless history**
and no longer needs in-place cleaning ("this can be messy"). Concretely, this session's §4d archiving was
**un-retired**:

- **P11/P13/P40 moved back into `PROBLEMS-AND-FIXES.md`** at their original slots (verbatim);
  `PROBLEMS-AND-FIXES-ARCHIVE.md` **deleted**.
- The three archiving-support edits were reverted: the P52-merge paragraph, the main-doc header pointer to
  the archive, and the STATUS.md MOSS "superseded → archive" marker (all referenced a file that no longer
  exists). Prior-session P15/P16/STATUS work was preserved (not touched).
- **KEPT:** the §4e staleness fixes (7 stale pointers-to-live-code) stay in `PROBLEMS-AND-FIXES.md` — those
  are accuracy corrections, not archiving.

**Lesson for the corpus:** the multi-session "retire/mark-superseded in place" method (30th–33rd sessions)
was retired in favour of **one clean current-state file + the raw P-log left messy**. Don't re-open the
archive idea. If you touch the clean map, the source of live truth is `PROBLEMS-AND-FIXES-CLEAN.md`; the
P-log and its `(full story: P##)` pointers are lossless history.

## 5. LEFT FOR NEXT SESSION (the user's call on each)

1. **~~`.env` lines ~126–130~~ — DONE (§4d), user-approved rewrite.**
2. **~~P11~~ — DONE (§4d), archived.**
3. **~~P40~~ — DONE (§4d): merged into P52, then archived.**
4. **P3 / P33 / P39 — recommended KEEP** (see §2 receipts). Reopen only with new evidence.
5. **Nothing is committed.** Branch `chore/cleanup-and-tts-merge`, which already carries 13 unpushed
   commits from earlier sessions. Decide commit/push with the user. **Uncommitted this session:** the
   4 files in §4d.

## 6. Standing cautions (carried from the repo's own rules)

- **The eye is the arbiter, both ways** (P19 + P33). A green probe does not clear anything touching the
  mouth/first-viseme; a *measured* delta is not automatically a *perceived* one either.
- **The avatar server is single-client** — close any `/studio` tab before driving a probe.
- Restart order after any avatar work: **CosyVoice (WSL) → MuseTalk → pipeline** — still the habit, but as
  of 2026-07-17 it is **insurance, not a requirement** (P15 §2; required again only above util ~0.75).
- **A refutation buried in a section nobody re-reads does not propagate.** P30 refuted P15 in 2026-07-05,
  in the same file, and 6 documents kept asserting the dead claim anyway. When you overturn a claim, `grep`
  the CLAIM'S WORDS across the repo the same day — not just the section number (the 30th session learned the
  section-number half of this; this is the other half).
