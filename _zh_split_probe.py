"""Why does zh synthesis yield nothing? text_normalize -> split_paragraph returns []?

v2 (baseline code, v2 weights) returns 0 bytes for EVERY zh input on /tts and /tts/stream,
with no 'synthesis text' log line at all -- meaning inference_zero_shot's
`for i in tqdm(self.frontend.text_normalize(...))` iterated zero times. English is fine.

Run in WSL:  conda activate cosyvllm && python _zh_split_probe.py
"""
import sys
import pathlib
from functools import partial

ROOT = pathlib.Path(__file__).parent / "CosyVoice"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "Matcha-TTS"))

from cosyvoice.utils.frontend_utils import contains_chinese, split_paragraph  # noqa: E402
from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer  # noqa: E402

TESTS = [
    "今天台北天氣晴朗。",
    "今天台北天氣晴朗，氣溫大約二十八度。",
    "今天台北天氣晴朗，氣溫大約二十八度，午後山區有短暫陣雨，外出記得攜帶雨具。",
]


def main():
    tok = get_qwen_tokenizer(
        token_path=str(ROOT / "pretrained_models" / "CosyVoice2-0.5B" / "CosyVoice-BlankEN"),
        skip_special_tokens=True,
    )
    enc = partial(tok.encode, allowed_special="all")

    for t in TESTS:
        n_tok = len(enc(t))
        segs = list(split_paragraph(enc, t, "zh", token_max_n=80, token_min_n=60,
                                    merge_len=20, comma_split=False)) if False else None
        # call with the SAME argument order cosyvoice/cli/frontend.py:151 uses
        segs = list(split_paragraph(t, enc, "zh", token_max_n=80, token_min_n=60,
                                    merge_len=20, comma_split=False))
        print(f"chinese={contains_chinese(t)} tokens={n_tok} segments={len(segs)} -> {segs!r}")


if __name__ == "__main__":
    main()
