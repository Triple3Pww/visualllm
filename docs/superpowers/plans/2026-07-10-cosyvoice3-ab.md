# CosyVoice 3 A/B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether `Fun-CosyVoice3-0.5B-2512` can replace `CosyVoice2-0.5B` as the pipeline TTS baseline, by gating on first-chunk TTFB under real MuseTalk render before spending any of the user's attention on quality.

**Architecture:** One WSL TTS server at `:8001`, switched between model dirs by the existing `COSYVOICE_MODEL_DIR` env var. `tts_engine.py` swaps its hardcoded `CosyVoice2(...)` for `AutoModel(...)`, which dispatches on the yaml present in the model dir. Arms run sequentially as A,A (null test) then A,B,A,B, with MuseTalk rendering a fixed WAV underneath to hold GPU contention constant. Probes are the existing `_ttfb_variance.py` and `_zh_audio_ab.py`.

**Tech Stack:** Python 3.10 (WSL `cosyvllm` conda env), vLLM, CosyVoice (vendored at `cosy3_pr` merge `ace7c47`), PyTorch flow decoder, MuseTalk (`musetalk` conda env on Windows).

**Spec:** `docs/superpowers/specs/2026-07-10-cosyvoice3-ab-design.md`

## Global Constraints

- **Two repos.** Engine + probe changes land in `E:\Claude\cosyvoice-local-tts` (its own git repo). Plan/spec/docs land in `E:\Claude\visualllm` on branch `exp/cosyvoice3-ab`. Commit in the repo you touched.
- **The probe may KILL, never APPROVE.** A passing gate promotes to phase 2 (user's eye); it never concludes "v3 is better." (P19/P22/P33.)
- **Both arms identical except the model dir:** same `pro_ref.wav` reference clip, same `COSYVOICE_VLLM_EAGER=1`, same `COSYVOICE_VLLM_GPU_UTIL`, same `COSYVOICE_FIRST_HOP_ZH=0`, same `COSYVOICE_FLOW_TRT=0`. If v3 forces a `gpu_util` change, re-run **both** arms at the new value or the run is void.
- **Load order every cycle:** CosyVoice (vLLM) up first, then MuseTalk. (P15.)
- **Never `localhost`** for `:8001` — always the WSL IP (`wsl hostname -I`). The WSL2 relay buffers the stream and fakes TTFB.
- **Do not modify** the live `.env`, `pipeline/metrics.py`, or the `CosyVoice/` submodule source.
- Gate thresholds: FAIL if v3 median TTFB > v2 median + 0.30s, or v3 max > 4.0s. v2 reference: median ~1.94s, max ~3.43s (P32, eager, under real render).
- Null test pass: the two v2 arms' medians differ by ≤ 0.15s.

---

## File Structure

| File | Repo | Responsibility |
|---|---|---|
| `tts_engine.py` | cosyvoice-local-tts | Model construction. Change: `CosyVoice2(...)` → `AutoModel(...)`. |
| `test_engine_dispatch.py` | cosyvoice-local-tts | **Create.** Unit test that `TTSEngine` dispatches via `AutoModel` and passes no `load_jit`. |
| `_zh_audio_ab.py` | cosyvoice-local-tts | Add **leading**-silence metric (P34 breath check). |
| `test_zh_audio_ab.py` | cosyvoice-local-tts | **Create.** Unit test for the leading-silence metric. |
| `_ab_run.py` | cosyvoice-local-tts | **Create.** Driver: run both probes against a live `:8001`, append a tagged record to `output/cv3_ab.json`. |

`_ab_run.py` deliberately does **not** start/stop servers. Process lifecycle on this box is
unreliable under load (see the `feedback-windows-process-tools-slow-under-load` memory: taskkill
and PowerShell hang for tens of seconds). The operator brings the arm up; the driver measures it.

---

### Task 1: `AutoModel` dispatch in `tts_engine.py`

**Files:**
- Modify: `E:\Claude\cosyvoice-local-tts\tts_engine.py:65` and `:83-84`
- Test: `E:\Claude\cosyvoice-local-tts\test_engine_dispatch.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `TTSEngine(model_dir=None, fp16=False, load_vllm=False, load_trt=False)` now constructs via `cosyvoice.cli.cosyvoice.AutoModel(model_dir=..., load_trt=..., load_vllm=..., fp16=...)`. `COSYVOICE_MODEL_DIR` already exists at `tts_engine.py:67` and needs no change.

**Why no `load_jit`:** `CosyVoice3.__init__` has no `load_jit` parameter (`cosyvoice/cli/cosyvoice.py:191`). `CosyVoice2` defaults it to `False`, which is what we pass today, so dropping it is behavior-preserving for v2.

- [ ] **Step 1: Write the failing test**

Create `E:\Claude\cosyvoice-local-tts\test_engine_dispatch.py`:

```python
"""TTSEngine must build the model via AutoModel (yaml-dispatched), not a hardcoded CosyVoice2.

This is what lets one server serve both CosyVoice2-0.5B and Fun-CosyVoice3-0.5B by
COSYVOICE_MODEL_DIR alone. CosyVoice3.__init__ takes no load_jit, so we must not pass it.
"""
import sys
import types
from unittest import mock

import pytest


@pytest.fixture
def fake_cosyvoice(monkeypatch, tmp_path):
    """Stub out torch + cosyvoice.cli.cosyvoice so no GPU/weights are needed."""
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    auto_model = mock.MagicMock()
    auto_model.return_value.sample_rate = 24000
    auto_model.return_value.add_zero_shot_spk.return_value = True
    mod = types.ModuleType("cosyvoice.cli.cosyvoice")
    mod.AutoModel = auto_model
    monkeypatch.setitem(sys.modules, "cosyvoice.cli.cosyvoice", mod)
    return auto_model


def test_engine_dispatches_via_automodel(fake_cosyvoice, tmp_path, monkeypatch):
    model_dir = tmp_path / "Fun-CosyVoice3-0.5B-2512"
    model_dir.mkdir()
    monkeypatch.setenv("COSYVOICE_MODEL_DIR", str(model_dir))

    import importlib
    import tts_engine
    importlib.reload(tts_engine)
    tts_engine.TTSEngine(load_vllm=True)

    fake_cosyvoice.assert_called_once()
    kwargs = fake_cosyvoice.call_args.kwargs
    assert kwargs["model_dir"] == str(model_dir)
    assert kwargs["load_vllm"] is True
    assert "load_jit" not in kwargs, "CosyVoice3.__init__ takes no load_jit"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /e/Claude/cosyvoice-local-tts && python -m pytest test_engine_dispatch.py -v
```
Expected: FAIL — `AttributeError` / `ImportError` on `AutoModel`, because `tts_engine` imports `CosyVoice2`.

- [ ] **Step 3: Write the minimal implementation**

In `tts_engine.py`, replace line 65:

```python
        from cosyvoice.cli.cosyvoice import AutoModel
```

and replace lines 78-84 with:

```python
        logging.info("Loading CosyVoice from %s (device=%s)", self.model_dir, self.device)

        # load_vllm: swap the autoregressive LLM onto vLLM (the real fix for first-chunk latency
        # -- the LLM token-gen is the ~3s bottleneck). Off by default (COSYVOICE_VLLM=1 to enable);
        # the Windows server stays on the PyTorch path. Requires the vLLM env (Linux/WSL).
        # AutoModel dispatches on the yaml in model_dir, so COSYVOICE_MODEL_DIR alone selects
        # CosyVoice2 vs CosyVoice3. No load_jit: CosyVoice3.__init__ has no such parameter, and
        # we passed False for it anyway.
        self.model = AutoModel(model_dir=self.model_dir, load_trt=load_trt,
                               load_vllm=load_vllm, fp16=fp16)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /e/Claude/cosyvoice-local-tts && python -m pytest test_engine_dispatch.py -v
```
Expected: PASS.

- [ ] **Step 5: Regression-check the v2 path still loads (no GPU needed for the import)**

```bash
cd /e/Claude/cosyvoice-local-tts && python -c "import tts_engine; print(tts_engine.DEFAULT_MODEL_DIR)"
```
Expected: prints the `CosyVoice2-0.5B` path, no traceback.

- [ ] **Step 6: Commit**

```bash
cd /e/Claude/cosyvoice-local-tts
git add tts_engine.py test_engine_dispatch.py
git commit -m "feat(engine): dispatch model via AutoModel so COSYVOICE_MODEL_DIR selects v2 or v3"
```

---

### Task 2: Leading-silence metric in `_zh_audio_ab.py`

**Files:**
- Modify: `E:\Claude\cosyvoice-local-tts\_zh_audio_ab.py:29-51` (`analyze`) and the print loop at `:62-72`
- Test: `E:\Claude\cosyvoice-local-tts\test_zh_audio_ab.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `analyze(wav_bytes) -> (duration_s, longest_internal_silence_s, silence_fraction, leading_silence_s)` — a 4-tuple. **Task 5 depends on this exact order.**

**Why:** P34 established that CosyVoice prepends a 25–610ms breath before the first zh word, and
the avatar lip-syncs off a Whisper of the waveform, so the mouth moves over it. v3's new tokenizer
may drop it. The probe already walks 20ms windows; leading silence is the count of silent windows
before the first energetic one — free to compute.

- [ ] **Step 1: Write the failing test**

Create `E:\Claude\cosyvoice-local-tts\test_zh_audio_ab.py`:

```python
"""analyze() must report LEADING silence (the P34 breath), not just internal silence."""
import array
import io
import wave

from _zh_audio_ab import analyze

SR = 16000


def _wav(samples):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(array.array("h", samples).tobytes())
    return buf.getvalue()


def test_leading_silence_measured():
    # 0.5s of silence, then 1.0s of loud tone.
    silence = [0] * int(0.5 * SR)
    loud = [20000 if i % 2 else -20000 for i in range(int(1.0 * SR))]
    dur, longest, frac, leading = analyze(_wav(silence + loud))

    assert 0.45 <= leading <= 0.55, f"leading={leading}"
    assert 1.4 <= dur <= 1.6


def test_no_leading_silence_when_speech_starts_immediately():
    loud = [20000 if i % 2 else -20000 for i in range(int(1.0 * SR))]
    _, _, _, leading = analyze(_wav(loud))
    assert leading < 0.05, f"leading={leading}"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /e/Claude/cosyvoice-local-tts && python -m pytest test_zh_audio_ab.py -v
```
Expected: FAIL — `ValueError: not enough values to unpack (expected 4, got 3)`.

- [ ] **Step 3: Write the minimal implementation**

In `_zh_audio_ab.py`, replace the body of `analyze` from the `longest = cur = 0` line through the
`return` with:

```python
    longest = cur = 0
    silent_wins = 0
    leading_wins = 0
    seen_speech = False
    for i in range(0, len(a) - win, win):
        peak = max(abs(x) for x in a[i:i + win])
        if peak < thresh:
            cur += 1
            longest = max(longest, cur)
            silent_wins += 1
            if not seen_speech:
                leading_wins += 1
        else:
            cur = 0
            seen_speech = True
    return (dur, longest * win / sr,
            silent_wins * win / sr / dur if dur else 0,
            leading_wins * win / sr)
```

Update the docstring's `Return` line to name the 4-tuple, and update the caller at `:64-69`:

```python
        dur, longest_sil, sil_frac, leading = analyze(wav)
        durs.append(dur)
        sils.append(longest_sil)
        leads.append(leading)
        flag = "  <-- LONG INTERNAL SILENCE" if longest_sil > 0.6 else ""
        print(f"  run{i}: dur={dur:5.2f}s  longest_silence={longest_sil:4.2f}s  "
              f"silence_frac={sil_frac:0.2f}  leading={leading:4.2f}s{flag}")
```

Initialize `leads = []` next to `durs, sils = [], []` at `:61`, and add after the MAX-SILENCE print:

```python
    print(f"LEADING-SILENCE median={statistics.median(leads):.2f}s  worst={max(leads):.2f}s  (P34 breath)")
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /e/Claude/cosyvoice-local-tts && python -m pytest test_zh_audio_ab.py -v
```
Expected: PASS, 2 tests.

- [ ] **Step 5: Commit**

```bash
cd /e/Claude/cosyvoice-local-tts
git add _zh_audio_ab.py test_zh_audio_ab.py
git commit -m "feat(probe): report leading silence (P34 breath) in _zh_audio_ab"
```

---

### Task 3: Fetch the v3 weights and verify the on-disk layout

**Files:**
- Create: `E:\Claude\cosyvoice-local-tts\CosyVoice\pretrained_models\Fun-CosyVoice3-0.5B-2512\`

**Interfaces:**
- Produces: a model dir containing `cosyvoice3.yaml`, `speech_tokenizer_v3.onnx`, `llm.pt`, `flow.pt`, `hift.pt`, `campplus.onnx`, `spk2info.pt`, `CosyVoice-BlankEN/`.

**This task can fail the whole plan.** `CosyVoice3.__init__` (`cosyvoice/cli/cosyvoice.py:196-207`)
requires exactly those filenames. If the HF repo ships a different layout, **stop and report** —
do not improvise a rename; a mismatched tokenizer would silently produce garbage.

- [ ] **Step 1: Check free disk (need ~7GB)**

```bash
df -h /e | tail -1
```
Expected: ≥ 10G available.

- [ ] **Step 2: Download**

```bash
cd /e/Claude/cosyvoice-local-tts && python -c "
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
                  local_dir='CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512')
"
```

- [ ] **Step 3: Verify the layout matches what `CosyVoice3.__init__` opens**

```bash
cd /e/Claude/cosyvoice-local-tts/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512 && \
  for f in cosyvoice3.yaml speech_tokenizer_v3.onnx llm.pt flow.pt hift.pt campplus.onnx spk2info.pt; do
    test -e "$f" && echo "OK   $f" || echo "MISS $f"
  done; test -d CosyVoice-BlankEN && echo "OK   CosyVoice-BlankEN/" || echo "MISS CosyVoice-BlankEN/"
```
Expected: all `OK`. **Any `MISS` → stop, report to the user, do not proceed.**

- [ ] **Step 4: Record the outcome (no commit — weights are gitignored)**

Confirm the weights dir is ignored:
```bash
cd /e/Claude/cosyvoice-local-tts && git status --porcelain | grep -c Fun-CosyVoice3 || echo "0 (ignored, good)"
```
Expected: `0`. If the weights show up as untracked, add `pretrained_models/` to `.gitignore` and commit that.

---

### Task 4: v3 smoke load — the three blocker checks

**Files:** none created. This task runs the WSL server by hand and reads its log.

**Interfaces:**
- Consumes: Task 1 (`AutoModel` dispatch), Task 3 (weights).
- Produces: a go/no-go on whether v3 can run on the **production path** (vLLM + RAS).

**Why this is its own task:** the spec commits us to reporting a v3-on-PyTorch result as a
**blocked test**, not as "v3 is slow." That distinction has to be settled before any timing run.

**The `<|endofprompt|>` requirement needs no code change.** `CosyVoice3LM` asserts token `151646`
is in `prompt_text` (`cosyvoice/llm/llm.py:591`), and `<|endofprompt|>` is already in the
tokenizer's `allowed_special` (`cosyvoice/tokenizer/tokenizer.py:249`). `prompt_text` comes from
the `COSYVOICE_PROMPT_TEXT` env (`tts_engine.py:48`), so the v3 arm simply sets it with the token
appended.

- [ ] **Step 1: Launch v3 in WSL with the v3 env**

```bash
wsl -d Ubuntu -e bash -c '
  export COSYVOICE_MODEL_DIR=/mnt/e/Claude/cosyvoice-local-tts/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512
  export COSYVOICE_PROMPT_TEXT="你好，我是你的AI虚拟助手，很高兴见到你。今天天气不错，有什么我可以帮你的<|endofprompt|>"
  bash /mnt/e/Claude/cosyvoice-local-tts/run_vllm_server.sh
' 2>&1 | tee /e/Claude/cosyvoice-local-tts/logs/v3_smoke.log
```

- [ ] **Step 2: Blocker check A — did vLLM accept the v3 checkpoint?**

```bash
grep -iE "CosyVoice2ForCausalLM|architecture not supported|Available KV cache" /e/Claude/cosyvoice-local-tts/logs/v3_smoke.log
```
Expected: the arch registers and **"Available KV cache memory" is positive**.
- If "architecture not supported" or "No available memory for the cache blocks" → **STOP.** Report as
  a blocked test (or, for the KV-cache case, as a VRAM affordability finding per spec risk 4).

- [ ] **Step 3: Blocker check B — is RAS actually firing (not merely registered)?**

```bash
grep -iE "ras|RasLogitsProcessor" /e/Claude/cosyvoice-local-tts/logs/v3_smoke.log
```
Expected: evidence the processor is attached to the engine. If it registers but never fires, add a
one-line `logging.info` at the top of `RasLogitsProcessor.__call__` to confirm, and note in the
report that RAS coverage on v3 is **assumed** rather than observed.

- [ ] **Step 4: Blocker check C — does a zh synth actually return audio?**

```bash
WSL_IP=$(wsl hostname -I | awk '{print $1}')
curl -s -m 60 -X POST "http://$WSL_IP:8001/tts" -H 'Content-Type: application/json' \
  -d '{"text":"今天台北天氣晴朗，氣溫大約二十八度。","speed":1.0}' -o /tmp/v3_smoke.wav
python -c "
import wave; w=wave.open('/tmp/v3_smoke.wav'); print('frames', w.getnframes(), 'sr', w.getframerate())
assert w.getnframes() > 8000, 'suspiciously short'
"
```
Expected: a multi-second wav. An `<|endofprompt|>` AssertionError here means the env from Step 1
didn't reach the engine.

- [ ] **Step 5: Record findings in the plan file, no code commit**

Append a short "Task 4 outcome" note to this plan documenting: vLLM yes/no, RAS observed/assumed,
`gpu_util` used, and any deviation. **If `gpu_util` had to change, Task 6 must re-run v2 at the
same value.**

---

### Task 5: The `_ab_run.py` driver

**Files:**
- Create: `E:\Claude\cosyvoice-local-tts\_ab_run.py`

**Interfaces:**
- Consumes: `analyze()` 4-tuple from Task 2; `ttfb()` from `_ttfb_variance.py`.
- Produces: `output/cv3_ab.json`, a list of records
  `{"tag": str, "cycle": int, "ttfb": [float], "zh": [{"dur","longest_sil","lead"}], "ts": float}`.

The driver measures a **live** `:8001`; it never starts or stops servers (Windows process tools
hang under load — the `feedback-windows-process-tools-slow-under-load` memory).

- [ ] **Step 1: Write the driver**

Create `E:\Claude\cosyvoice-local-tts\_ab_run.py`:

```python
"""One arm of the CosyVoice2-vs-3 A/B: measure the LIVE :8001 server, append a tagged record.

The operator brings the arm up (COSYVOICE_MODEL_DIR selects v2 or v3) and runs this once per
cycle. Sequential arms mean thermal/background drift is a confound, so we run A,B,A,B and
compare medians WITHIN a cycle -- never a single A against a single B.

  python _ab_run.py --host <WSL_IP> --tag v2 --cycle 1
"""
import argparse
import json
import pathlib
import statistics
import time

from _ttfb_variance import OPENERS, ttfb
from _zh_audio_ab import ZH, analyze, synth

OUT = pathlib.Path(__file__).parent / "output" / "cv3_ab.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="WSL IP, NOT localhost")
    ap.add_argument("--tag", required=True, choices=["v2", "v3"])
    ap.add_argument("--cycle", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--zh-runs", type=int, default=5)
    args = ap.parse_args()

    ttfb(args.host, "warm the socket")  # not counted

    ttfbs = []
    for r in range(args.rounds):
        for name, text in OPENERS:
            t = ttfb(args.host, text)
            ttfbs.append(t)
            print(f"  c{args.cycle} {args.tag} r{r+1} {name}: {t:.3f}s")

    zh = []
    for i in range(args.zh_runs):
        dur, longest, _frac, lead = analyze(synth(args.host, ZH))
        zh.append({"dur": dur, "longest_sil": longest, "lead": lead})
        print(f"  c{args.cycle} {args.tag} zh{i+1}: dur={dur:.2f}s lead={lead:.2f}s")

    rec = {"tag": args.tag, "cycle": args.cycle, "ttfb": ttfbs, "zh": zh, "ts": time.time()}
    OUT.parent.mkdir(exist_ok=True)
    records = json.loads(OUT.read_text()) if OUT.exists() else []
    records.append(rec)
    OUT.write_text(json.dumps(records, indent=2))

    print(f"\n{args.tag} cycle{args.cycle}  n={len(ttfbs)}  "
          f"median={statistics.median(ttfbs):.3f}  max={max(ttfbs):.3f}  "
          f"stddev={statistics.pstdev(ttfbs):.3f}")
    print(f"  zh lead median={statistics.median(x['lead'] for x in zh):.2f}s")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it imports cleanly (no server needed)**

```bash
cd /e/Claude/cosyvoice-local-tts && python -c "import _ab_run; print('ok')"
```
Expected: `ok`. A failure here means Task 2 changed `analyze`'s arity but not its callers.

- [ ] **Step 3: Commit**

```bash
cd /e/Claude/cosyvoice-local-tts
git add _ab_run.py
git commit -m "feat(probe): _ab_run driver -- one tagged arm into output/cv3_ab.json"
```

---

### Task 6: The null test (A,A) — prove the rig measures models, not drift

**Files:** none. Produces `output/cv3_ab.json` records with `tag=v2, cycle=0` twice.

**Interfaces:**
- Consumes: Task 5's driver.
- Produces: a go/no-go on whether any A/B number from this rig is trustworthy.

**Run both arms as v2.** If they disagree by more than 0.15s median, the rig is measuring drift.

- [ ] **Step 1: Bring up the contention rig (load order matters — P15)**

Terminal 1 — CosyVoice v2 in WSL (near-empty card):
```bash
wsl -d Ubuntu -e bash -c "bash /mnt/e/Claude/cosyvoice-local-tts/run_vllm_server.sh"
```

Terminal 2 — MuseTalk server, production env:
```bash
E:\miniconda3\envs\musetalk\python.exe -u -m local_services.musetalk_server.app
```

Terminal 3 — hold GPU contention constant by looping a fixed WAV through the renderer:
```bash
cd /e/Claude/visualllm && while true; do python -m scripts._drive_frames output/reply_concise.wav 12; done
```

- [ ] **Step 2: Arm 1 (v2)**

```bash
WSL_IP=$(wsl hostname -I | awk '{print $1}')
cd /e/Claude/cosyvoice-local-tts && python _ab_run.py --host $WSL_IP --tag v2 --cycle 0
```

- [ ] **Step 3: Restart the v2 server, then Arm 2 (also v2)**

Stop and relaunch the WSL server exactly as in Step 1 — the relaunch is part of what we're testing,
because the real A/B relaunches between arms.

```bash
WSL_IP=$(wsl hostname -I | awk '{print $1}')
cd /e/Claude/cosyvoice-local-tts && python _ab_run.py --host $WSL_IP --tag v2 --cycle 0
```

- [ ] **Step 4: Compare — this is the gate on the rig itself**

```bash
cd /e/Claude/cosyvoice-local-tts && python -c "
import json, statistics as s
recs = [r for r in json.load(open('output/cv3_ab.json')) if r['cycle'] == 0]
assert len(recs) == 2, f'expected 2 null-test records, got {len(recs)}'
m = [s.median(r['ttfb']) for r in recs]
print(f'v2 arm A median={m[0]:.3f}  v2 arm B median={m[1]:.3f}  delta={abs(m[0]-m[1]):.3f}s')
print('NULL TEST PASS' if abs(m[0]-m[1]) <= 0.15 else 'NULL TEST FAIL -- rig measures drift, STOP')
"
```
Expected: `NULL TEST PASS`. **On FAIL: stop and report.** Do not proceed to Task 7; every number
the rig produces would be noise.

---

### Task 7: The real A/B (A,B,A,B) and the gate verdict

**Files:** none. Appends `cycle=1` and `cycle=2` records.

**Interfaces:**
- Consumes: Tasks 4, 5, 6.
- Produces: a gate verdict — FAIL (report and stop) or PASS (promote to phase 2).

Keep the MuseTalk + `_drive_frames` contention rig from Task 6 running untouched across all four
arms. Only the WSL TTS server is relaunched.

- [ ] **Step 1: Cycle 1, arm A (v2)**

Relaunch the WSL server with no `COSYVOICE_MODEL_DIR` override (v2 default), then:
```bash
WSL_IP=$(wsl hostname -I | awk '{print $1}')
cd /e/Claude/cosyvoice-local-tts && python _ab_run.py --host $WSL_IP --tag v2 --cycle 1
```

- [ ] **Step 2: Cycle 1, arm B (v3)**

Relaunch the WSL server with the v3 env from Task 4 Step 1, then:
```bash
WSL_IP=$(wsl hostname -I | awk '{print $1}')
cd /e/Claude/cosyvoice-local-tts && python _ab_run.py --host $WSL_IP --tag v3 --cycle 1
```

- [ ] **Step 3: Cycle 2 — repeat Steps 1 and 2 with `--cycle 2`**

Same commands, `--cycle 2`. Four arms total, alternating.

- [ ] **Step 4: Compute the verdict**

```bash
cd /e/Claude/cosyvoice-local-tts && python -c "
import json, statistics as s
recs = [r for r in json.load(open('output/cv3_ab.json')) if r['cycle'] in (1, 2)]
by = {}
for r in recs:
    by.setdefault((r['cycle'], r['tag']), []).extend(r['ttfb'])
for c in (1, 2):
    v2, v3 = by[(c, 'v2')], by[(c, 'v3')]
    print(f'cycle{c}: v2 med={s.median(v2):.3f} max={max(v2):.3f} | '
          f'v3 med={s.median(v3):.3f} max={max(v3):.3f} | delta={s.median(v3)-s.median(v2):+.3f}s')
allv2 = [t for (c, tag), ts in by.items() if tag == 'v2' for t in ts]
allv3 = [t for (c, tag), ts in by.items() if tag == 'v3' for t in ts]
d, mx = s.median(allv3) - s.median(allv2), max(allv3)
print(f'\nOVERALL delta={d:+.3f}s  v3 max={mx:.3f}s')
print('GATE FAIL' if (d > 0.30 or mx > 4.0) else 'GATE PASS -> promote to phase 2 (user watches clips)')
"
```

- [ ] **Step 5: Also report the zh leading-silence (P34) delta, whatever the gate says**

```bash
cd /e/Claude/cosyvoice-local-tts && python -c "
import json, statistics as s
recs = [r for r in json.load(open('output/cv3_ab.json')) if r['cycle'] in (1, 2)]
for tag in ('v2', 'v3'):
    leads = [z['lead'] for r in recs if r['tag'] == tag for z in r['zh']]
    sils  = [z['longest_sil'] for r in recs if r['tag'] == tag for z in r['zh']]
    print(f'{tag}: leading-silence median={s.median(leads):.2f}s  longest-internal median={s.median(sils):.2f}s')
"
```

This is free information regardless of the verdict: if v3 drops the P34 breath, that is worth
knowing even if v3 fails the latency gate.

- [ ] **Step 6: Report to the user**

Write the numbers, the verdict, and any Task 4 deviation. **On GATE FAIL, stop here** — do not
render clips. On GATE PASS, present phase 2 (Task 8) as the next step and let the user decide when.

---

### Task 8: Phase 2 — matched zh clips for the user's eye (only on GATE PASS)

**Files:** produces two mp4s under `E:\Claude\visualllm\output\`.

**Interfaces:**
- Consumes: a GATE PASS from Task 7.

**P40's rule governs this task:** the reference must not share the suspect input. Render from the
**delivered** frames (`MUSETALK_DUMP_DELIVERED`) — what the browser actually gets — never an offline
render fed a repaired PCM copy, which bypasses the path under test and always looks good.

- [ ] **Step 1: Capture v2**

Bring up the v2 arm plus the full pipeline, set `MUSETALK_DUMP_DELIVERED=1`, drive one zh turn,
and keep the delivered-frame dump as `output/zh_v2.mp4`.

- [ ] **Step 2: Capture v3**

Same, with the v3 env from Task 4 Step 1 → `output/zh_v3.mp4`.

- [ ] **Step 3: Same text, same reference clip, both arms**

Verify from the logs that both used `pro_ref.wav` and the identical zh sentence. If not, re-capture:
per P18, varying the reference clip means A/B-ing the voice rather than the architecture.

- [ ] **Step 4: Hand the two files to the user**

Do not summarize them as "v3 looks better." Per the spec's verdict rule and the
`feedback-test-fix-before-handing` memory: hand over two files and let the eye decide.

---

## Self-Review

**Spec coverage.** `AutoModel` dispatch → Task 1. Weights → Task 3. Contention rig → Task 6 Step 1.
Probes + leading silence → Tasks 2, 5. Null test → Task 6. A,B,A,B → Task 7. Gate thresholds →
Task 7 Step 4. Controls (same ref clip, same gpu_util, load order) → Global Constraints + Task 4
Step 5. Risks 1–4 → Task 4 Steps 2–4 (risk 1 resolved: env-only, no code change). Rejected
alternatives and out-of-scope items → no tasks, correctly. Phase 2 → Task 8.

**Type consistency.** `analyze()` returns a 4-tuple `(dur, longest_sil, frac, leading)` in Task 2;
Task 5's driver unpacks exactly that arity and ignores `frac` as `_frac`. `ttfb(host, text)` and
`OPENERS` are imported from `_ttfb_variance.py` unchanged. `synth(host, text)` and `ZH` come from
`_zh_audio_ab.py` unchanged.

**Gap found and closed:** the spec said `COSYVOICE_MODEL_DIR` was a *new* env var; it already
exists at `tts_engine.py:67`. Task 1 reflects reality (only the `CosyVoice2` hardcode changes).
