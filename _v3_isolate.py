"""Isolate: does CosyVoice3 produce correct-length zh audio WITHOUT vLLM (and thus without
our RasLogitsProcessor)? The vLLM+RAS server emits ~1.0s for a sentence that should be ~4-5s.

RAS was written against CosyVoice2's speech-token vocabulary; CosyVoice3LM has different
sos/task/eos ids. If PyTorch is correct and vLLM is short, RAS (or the vLLM wrapper) is the
culprit and the "RAS carries over free" assumption in the spec is wrong.

Run in WSL:  conda activate cosyvllm && python _v3_isolate.py
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).parent / "CosyVoice"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "Matcha-TTS"))

import torch  # noqa: E402
import torchaudio  # noqa: E402
from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402

MD = str(ROOT / "pretrained_models" / "Fun-CosyVoice3-0.5B-2512")
REF = str(ROOT / "asset" / "pro_ref.wav")
PT = ("You are a helpful assistant.<|endofprompt|>"
      "你好，我是你的AI虚拟助手，很高兴见到你。今天天气不错，有什么我可以帮你的")
ZH = "今天台北天氣晴朗，氣溫大約二十八度。"
EN = "Sure, let me check that for you."
V3_INSTRUCT = "You are a helpful assistant.<|endofprompt|>"


def main():
    m = AutoModel(model_dir=MD, load_vllm=False, load_trt=False, fp16=False)
    sr = m.sample_rate

    # Mirror production (tts_engine.py): register the reference by PATH, then synth by spk id.
    ok = m.add_zero_shot_spk(PT, REF, "probe")
    print(f"add_zero_shot_spk -> {ok}", flush=True)

    chunks = [o["tts_speech"] for o in m.inference_zero_shot(ZH, "", "", zero_shot_spk_id="probe", stream=False)]
    w = torch.concat(chunks, dim=1)
    print(f"RESULT pytorch zero_shot zh: dur={w.shape[1]/sr:.2f}s  (expect ~4-5s)", flush=True)
    torchaudio.save(str(pathlib.Path(__file__).parent / "output" / "v3_pytorch_zh.wav"), w, sr)

    chunks = [o["tts_speech"] for o in m.inference_cross_lingual(V3_INSTRUCT + EN, REF, zero_shot_spk_id="probe", stream=False)]
    w = torch.concat(chunks, dim=1)
    print(f"RESULT pytorch cross_lingual en: dur={w.shape[1]/sr:.2f}s  (expect ~2s)", flush=True)


if __name__ == "__main__":
    main()
