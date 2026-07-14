"""Build + validate the flow-matching fp16 TensorRT engine (Lever 1).

Loads CosyVoice2 with COSYVOICE_FLOW_TRT=1 (load_trt=True, fp16=True) so CosyVoice
auto-builds flow.decoder.estimator.fp16.mygpu.plan from the shipped fp32 onnx, then
synthesizes en+zh to confirm the fp16 TRT estimator produces clean audio (not NaN/silence).
Run in WSL cosyvllm env with the vLLM server STOPPED (needs the ~4GB TRT build workspace).
"""
import os, time
os.environ["COSYVOICE_FLOW_TRT"] = "1"
os.environ.setdefault("COSYVOICE_VLLM", "0")  # build w/o vLLM to save VRAM for the TRT workspace

import torchaudio
from tts_engine import get_engine

t0 = time.time()
eng = get_engine()
print(f"[build] load+build took {time.time()-t0:.1f}s")
print(f"[build] flow estimator now: {type(eng.model.model.flow.decoder.estimator).__name__}")

for name, txt in [("en", "Today's weather in Taipei is sunny and mild, with a gentle breeze."),
                  ("zh", "今天台北天氣晴朗，氣溫二十八度，適合出門走走。")]:
    t0 = time.time()
    wav, sr = eng.synthesize(txt)
    dur = wav.shape[1] / sr
    g = time.time() - t0
    peak = wav.abs().max().item()
    print(f"[synth] {name}: gen {g:.2f}s  audio {dur:.2f}s  rtf {g/dur:.2f}  peak {peak:.3f}"
          f"  {'OK' if 0.05 < peak <= 1.01 else 'SUSPECT (silent/clipped/NaN)'}")
    torchaudio.save(f"/mnt/e/Claude/VisualLLm/tts/cosyvoice-server/outputs/_flowtrt_{name}.wav", wav, sr)
print("[build] done")
