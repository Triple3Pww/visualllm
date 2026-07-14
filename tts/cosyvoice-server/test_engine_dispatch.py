"""TTSEngine must build the model via AutoModel (yaml-dispatched), not a hardcoded CosyVoice2.

This is what lets one server serve both CosyVoice2-0.5B and Fun-CosyVoice3-0.5B by
COSYVOICE_MODEL_DIR alone. CosyVoice3.__init__ takes no load_jit, so we must not pass it.
"""
import sys
import types
from unittest import mock

import pytest


@pytest.fixture
def fake_automodel(monkeypatch):
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


def test_engine_dispatches_via_automodel(fake_automodel, tmp_path, monkeypatch):
    model_dir = tmp_path / "Fun-CosyVoice3-0.5B-2512"
    model_dir.mkdir()
    monkeypatch.setenv("COSYVOICE_MODEL_DIR", str(model_dir))

    import tts_engine
    tts_engine.TTSEngine(load_vllm=True)

    fake_automodel.assert_called_once()
    kwargs = fake_automodel.call_args.kwargs
    assert kwargs["model_dir"] == str(model_dir)
    assert kwargs["load_vllm"] is True
    assert "load_jit" not in kwargs, "CosyVoice3.__init__ takes no load_jit"


def _engine_with_model_class(fake_automodel, tmp_path, monkeypatch, class_name):
    """Build a TTSEngine whose underlying model reports the given class name."""
    model_dir = tmp_path / class_name
    model_dir.mkdir()
    monkeypatch.setenv("COSYVOICE_MODEL_DIR", str(model_dir))
    fake_automodel.return_value.__class__ = type(class_name, (), {})

    import tts_engine
    eng = tts_engine.TTSEngine()
    eng.model.inference_cross_lingual.return_value = iter(())
    return eng


def test_v3_prepends_instruct_prefix_to_english_text(fake_automodel, tmp_path, monkeypatch):
    """CosyVoice3 asserts <|endofprompt|> is in the tokens the LM sees, and SPLITS on it:
    prefix before, real text after. cross_lingual (English) deletes prompt_text, so the whole
    'You are a helpful assistant.<|endofprompt|>' prefix must ride on the text -- matching
    upstream's cosyvoice3_example (example.py:81). A bare marker is not the documented form."""
    eng = _engine_with_model_class(fake_automodel, tmp_path, monkeypatch, "CosyVoice3")
    list(eng.synthesize_stream("Hello, warming up."))

    sent = eng.model.inference_cross_lingual.call_args.args[0]
    assert sent == "You are a helpful assistant.<|endofprompt|>Hello, warming up.", sent


def test_v2_english_text_is_untouched(fake_automodel, tmp_path, monkeypatch):
    eng = _engine_with_model_class(fake_automodel, tmp_path, monkeypatch, "CosyVoice2")
    list(eng.synthesize_stream("Hello, warming up."))

    sent = eng.model.inference_cross_lingual.call_args.args[0]
    assert sent == "Hello, warming up.", sent


def test_v3_chinese_text_is_untouched(fake_automodel, tmp_path, monkeypatch):
    """zh uses inference_zero_shot, which keeps prompt_text -- the marker is already there."""
    eng = _engine_with_model_class(fake_automodel, tmp_path, monkeypatch, "CosyVoice3")
    eng.model.inference_zero_shot.return_value = iter(())
    list(eng.synthesize_stream("今天天氣晴朗。"))

    sent = eng.model.inference_zero_shot.call_args.args[0]
    assert sent == "今天天氣晴朗。", sent
