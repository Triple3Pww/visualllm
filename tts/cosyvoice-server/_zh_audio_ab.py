"""Does enabling CUDA graphs change the zh AUDIO (the lipsync-mismatch report, 2026-07-05)?

The user sees zh lips not matching the words with graphs ON, fine with graphs OFF. Lips come
from MuseTalk's whisper of the audio, so if graphs degrade the zh audio (esp. RAS silence-loops,
P18) the mouth desyncs from the phonemes. This generates the SAME zh sentence N times over /tts
and reports, per run: audio duration and the LONGEST internal silence run (the RAS-loop artifact
= dead silence mid-sentence, which renders as a closed mouth while the voice should be speaking).

Run per config (flip EAGER + relaunch WSL server between):  python _zh_audio_ab.py --host <WSL_IP>
"""
import argparse
import io
import json
import statistics
import urllib.request
import wave

ZH = "今天台北天氣晴朗，氣溫大約二十八度，午後山區有短暫陣雨，外出記得攜帶雨具。"


def synth(host, text):
    body = json.dumps({"text": text, "speed": 1.0}).encode()
    req = urllib.request.Request(f"http://{host}:8001/tts", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def analyze(wav_bytes):
    """Return (duration_s, longest_internal_silence_s, silence_fraction, leading_silence_s).

    leading_silence_s = silence before the FIRST energetic window. That is the P34 breath:
    CosyVoice's zero-shot synth prepends a low-level breath before the first zh word, and
    MuseTalk lip-syncs off a Whisper of the waveform, so the mouth moves over it.
    """
    w = wave.open(io.BytesIO(wav_bytes))
    sr, n = w.getframerate(), w.getnframes()
    raw = w.readframes(n)
    import array
    a = array.array("h")
    a.frombytes(raw)
    dur = n / sr
    # 20ms windows; a window is "silent" if peak amplitude < 2% of full scale
    win = max(1, sr // 50)
    thresh = int(0.02 * 32768)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="172.24.44.238")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    print(f"host={args.host} runs={args.runs} tag={args.tag}  ({len(ZH)} zh chars)\n")
    durs, sils, leads = [], [], []
    for i in range(1, args.runs + 1):
        wav = synth(args.host, ZH)
        dur, longest_sil, sil_frac, leading = analyze(wav)
        durs.append(dur)
        sils.append(longest_sil)
        leads.append(leading)
        flag = "  <-- LONG INTERNAL SILENCE" if longest_sil > 0.6 else ""
        print(f"  run{i}: dur={dur:5.2f}s  longest_silence={longest_sil:4.2f}s  "
              f"silence_frac={sil_frac:0.2f}  leading={leading:4.2f}s{flag}")
    print(f"\nDURATION   median={statistics.median(durs):.2f}s  "
          f"min={min(durs):.2f}  max={max(durs):.2f}  spread={max(durs)-min(durs):.2f}s")
    print(f"MAX-SILENCE median={statistics.median(sils):.2f}s  worst={max(sils):.2f}s")
    print(f"LEADING-SILENCE median={statistics.median(leads):.2f}s  worst={max(leads):.2f}s  (P34 breath)")


if __name__ == "__main__":
    main()
