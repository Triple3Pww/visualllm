"""Synthesize speech from text with the project's ElevenLabs voice (offline helper).

Reads ELEVENLABS_* from the repo .env (same creds the pipeline uses) and writes an
mp3, then (if ffmpeg is on PATH) a 24 kHz mono wav alongside it -- the wav is what the
offline avatar renderer (local_services/ditto_offline.py) consumes.

    python -m scripts.tts_elevenlabs --text-file output/what_is_ai.txt --out output/what_is_ai.mp3
    python -m scripts.tts_elevenlabs --text "Hello there" --out output/hi.mp3
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / ".env"


def _load_env() -> dict:
    env: dict[str, str] = {}
    if ENV.exists():
        for line in ENV.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                v = v.strip()
                # Strip an inline comment (dotenv: a '#' preceded by whitespace), then
                # any surrounding quotes. The repo .env uses inline comments like
                # `ID=pNInz...   # Adam`, which must not leak into the value.
                m = re.search(r"\s#", v)
                if m:
                    v = v[: m.start()]
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def synth(text: str, out_mp3: Path) -> Path:
    env = _load_env()
    key = env.get("ELEVENLABS_API_KEY")
    voice = env.get("ELEVENLABS_VOICE_ID")
    model = env.get("ELEVENLABS_MODEL", "eleven_flash_v2_5")
    if not key or not voice:
        sys.exit("ELEVENLABS_API_KEY / ELEVENLABS_VOICE_ID missing from .env")

    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
        "?output_format=mp3_44100_128"
    )
    body = json.dumps({
        "text": text,
        "model_id": model,
        # A touch of style + speaker boost for a clear, lively narration.
        "voice_settings": {
            "stability": 0.45,
            "similarity_boost": 0.8,
            "style": 0.15,
            "use_speaker_boost": True,
        },
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"xi-api-key": key, "Content-Type": "application/json",
                 "Accept": "audio/mpeg"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        audio = r.read()
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    out_mp3.write_bytes(audio)
    print(f"wrote {out_mp3} ({len(audio)} bytes)")

    # Best-effort 24 kHz mono wav for the renderer (librosa can read mp3 too, but a
    # plain wav avoids any audioread backend surprises in the ditto env).
    out_wav = out_mp3.with_suffix(".wav")
    try:
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-i", str(out_mp3),
             "-ar", "24000", "-ac", "1", str(out_wav)],
            check=True,
        )
        print(f"wrote {out_wav}")
    except Exception as e:  # noqa: BLE001 -- wav is a convenience, mp3 is the source
        print(f"(skipped wav: {e})")
    return out_mp3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="text to speak")
    ap.add_argument("--text-file", help="file containing the text to speak")
    ap.add_argument("--out", required=True, help="output .mp3 path")
    args = ap.parse_args()

    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8").strip()
    elif args.text:
        text = args.text
    else:
        sys.exit("provide --text or --text-file")
    synth(text, Path(args.out))


if __name__ == "__main__":
    main()
