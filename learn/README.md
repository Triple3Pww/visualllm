# learn/ — the VisualLLm fundamentals course

**Open `index.html` in your browser. That's it.**

Four modules, one per week. Each follows one conversation turn through the system and
teaches the layer it crosses: audio -> async streaming -> model inference -> GPU and A/V sync.
Every module ends on a real bug from `docs/PROBLEMS-AND-FIXES.md` that you can now explain.

The page teaches and examines you. The `.py` files here are your lab — you open them in
your editor, fill the `TODO(you)` blanks, and run them:

    python learn/m1_vad.py output/q_ai.wav

Nothing here needs a GPU, a conda env, or an install. Plain Python 3.11 and a browser.

**The page holds no code, on purpose.** It names the file and the blank; the code lives only
in the `.py` file, so the two can never drift apart.

## Using an outside tutor

`COACH-BRIEF.md` is a self-contained teaching brief you can hand to another AI (Gemini) to be
taught a module in depth. **Order matters:** commit that module's cold pretest guesses in
`index.html` FIRST, then go to the coach, then come back for the lab and the self-check.
Guessing before you are taught is the mechanism — teach yourself first and you spend it for
nothing. The brief tells the coach the same thing, and tells it not to hand you the blanks.

Design + evidence: `docs/superpowers/specs/2026-07-11-visualllm-study-plan-design.md`
