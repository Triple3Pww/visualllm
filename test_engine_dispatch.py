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
