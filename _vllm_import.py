import sys
from pathlib import Path
H = Path("/mnt/e/Claude/cosyvoice-local-tts/CosyVoice")
for p in (str(H), str(H / "third_party" / "Matcha-TTS")):
    sys.path.insert(0, p)
import vllm
print("vllm", vllm.__version__)
try:
    from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
    print("CUSTOM_MODEL_IMPORT_OK")
    from vllm import ModelRegistry
    ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
    print("REGISTER_OK")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("FAILED:", repr(e))
