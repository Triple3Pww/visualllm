# assets

Put your **own portrait here** to use it as the local MuseTalk avatar — free,
no plan gate, runs on the 5060 Ti.

```
assets/avatar.png      <- your photo (front-facing, clear face)
```

The MuseTalk server reads this path from the `AVATAR_REF` env var
(default `assets/avatar.png`):

```
set AVATAR_REF=assets/avatar.png
python -m local_services.musetalk_server.app
```

Then in `.env`: `AVATAR_PROVIDER=musetalk_local`.

## Good portrait = good lip-sync
- Front-facing, single person, face large and clearly visible
- Neutral expression, mouth closed or slightly open
- Sharp, evenly lit, photoreal (not an illustration)
- A short forward-facing **video** also works and gives subtler motion than a
  still image

## Two ways to use your own face
| Route | Cost | Real-time? | Setup |
|-------|------|-----------|-------|
| **Local MuseTalk** (this folder) | free (your GPU) | yes (30 fps) | drop in `avatar.png`, run the server |
| **Simli custom face** | maybe paid (unconfirmed free) | yes, after one-time face creation | `python -m scripts.simli_create_face --image assets/avatar.png` → `SIMLI_FACE_ID` |

(Put the actual image file in this folder; it's git-ignored via `*.png` rules in
`.gitignore` — commit a placeholder separately if you need one in version control.)
