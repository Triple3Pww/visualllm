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
