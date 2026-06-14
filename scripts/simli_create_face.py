"""Create a custom Simli avatar face from YOUR OWN image -> prints a faceId.

This turns "use my own photo on Simli" into one command. It uploads your image
to Simli's face-generation endpoint; the returned ID goes in .env as
SIMLI_FACE_ID and the pipeline uses your face automatically.

Notes:
- Creation is a ONE-TIME prep step (Simli says it can take a while), not
  real-time. Run it once, save the ID.
- Free-tier eligibility for custom faces is not stated in Simli's docs — if the
  API rejects it on the free plan, that's the gate. (The free local alternative
  is MuseTalk: see local_services/musetalk_server + assets/README.md.)

Requires: `pip install requests`

Usage:
    python -m scripts.simli_create_face --image me.png --name my_avatar
    python -m scripts.simli_create_face --image me.png --legacy   # older model
"""
from __future__ import annotations

import argparse
import json
import os
import sys

TRINITY_URL = "https://api.simli.ai/faces/trinity"   # current model
LEGACY_URL = "https://api.simli.ai/generateFaceID"    # older model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path to your portrait (front-facing)")
    ap.add_argument("--name", default="my_avatar", help="name for the face")
    ap.add_argument("--legacy", action="store_true", help="use the legacy face model")
    ap.add_argument("--api-key", default=os.getenv("SIMLI_API_KEY"))
    args = ap.parse_args()

    if not args.api_key:
        print("ERROR: set SIMLI_API_KEY in .env or pass --api-key", file=sys.stderr)
        return 2
    if not os.path.isfile(args.image):
        print(f"ERROR: image not found: {args.image}", file=sys.stderr)
        return 2

    try:
        import requests
    except ImportError:
        print("ERROR: pip install requests", file=sys.stderr)
        return 2

    url = LEGACY_URL if args.legacy else TRINITY_URL
    headers = {"x-simli-api-key": args.api_key}
    params = {"face_name": args.name}

    print(f"Uploading {args.image} -> {url} (this can take a while)…")
    with open(args.image, "rb") as f:
        resp = requests.post(
            url, headers=headers, params=params,
            files={"image": (os.path.basename(args.image), f)},
            timeout=600,
        )

    print(f"HTTP {resp.status_code}")
    try:
        data = resp.json()
    except Exception:  # noqa: BLE001
        print(resp.text)
        return 1 if resp.status_code >= 400 else 0

    print(json.dumps(data, indent=2, ensure_ascii=False))
    if resp.status_code >= 400:
        print("\nRequest failed — if this is a plan/quota error, your free tier may "
              "not allow custom faces. Use a preset face, or the local MuseTalk path.",
              file=sys.stderr)
        return 1

    # Surface a faceId under whatever key Simli used.
    for key in ("faceId", "face_id", "id", "character_uid", "faceID"):
        if isinstance(data, dict) and data.get(key):
            print(f"\n==> SIMLI_FACE_ID={data[key]}")
            break
    else:
        print("\n(Could not auto-detect the face ID field — copy it from the JSON above.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
