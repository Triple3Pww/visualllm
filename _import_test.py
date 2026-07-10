import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
COSY = HERE / "CosyVoice"
MATCHA = COSY / "third_party" / "Matcha-TTS"
for p in (str(COSY), str(MATCHA)):
    if p not in sys.path:
        sys.path.insert(0, p)
import transformers
print("transformers", transformers.__version__)
try:
    from cosyvoice.cli.cosyvoice import CosyVoice2
    print("IMPORT_OK CosyVoice2")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("IMPORT_FAILED:", repr(e))
