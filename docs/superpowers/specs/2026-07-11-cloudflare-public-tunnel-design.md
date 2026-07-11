# Public access via Cloudflare Quick Tunnel (replaces Tailscale Funnel)

**Date:** 2026-07-11
**Status:** design approved, pending spec review
**Goal:** anyone with a link can reach the avatar, hosted on this box. Tailscale Funnel
stopped working (login-gated admin toggle, wiped config); replace the public front door with
a Cloudflare quick tunnel. Media stays on the already-shipped STUN path.

## Background / current state (verified on disk 2026-07-11)

- The public-access plumbing already exists on `main` (committed `1f9d3bb`):
  `_install_turn_ice_servers()` + `_restrict_ice_to_subnet()` in `pipeline/main.py`, gated by
  `WEBRTC_PUBLIC` / `TURN_URLS`. Currently **off** (`.env` `WEBRTC_PUBLIC=0`).
- Front door is back to `tailscale serve` (all three mappings show "tailnet only"; Funnel off).
- `cloudflared` v2025.8.1 is already installed at `C:\Program Files (x86)\cloudflared\cloudflared.exe`.
  No account/domain/cert configured (no `~/.cloudflared`), so a **named** tunnel is out; a **quick**
  tunnel needs none of that.
- `pipeline/config.py` calls `load_dotenv()`, so `python -m pipeline.main` reads `WEBRTC_PUBLIC`
  directly from `.env`. Flipping `.env` + restarting the pipeline is sufficient (this is exactly the
  path verified live with a real off-tailnet Android phone on 2026-07-09).

## The two independent layers (the load-bearing insight)

A tunnel (Cloudflare **or** Tailscale Funnel) only carries the **HTTPS page + `/api/offer`
signaling**. It does **not** carry WebRTC media (mic/avatar are UDP P2P over ICE). So the tunnel
only changes the *link*; the media still rides on STUN/TURN and is unaffected by how the page is
hosted.

1. **Front door (the link)** = Cloudflare quick tunnel -> a public `https://<random>.trycloudflare.com`.
2. **Media path** = `WEBRTC_PUBLIC=1` -> the server gathers a STUN srflx candidate (public
   `171.99.155.119`) and `_restrict_ice_to_subnet` pins ICE to `{Tailscale 100.64/10 +
   default-route /32}`. This box's NAT is port-preserving (cone), so **STUN-only works, no TURN**.

## Decisions (all approved)

- **Quick tunnel** (random URL, zero account/domain) — not a named tunnel.
- **Auto-start with `launch.ps1`** (the one-click `.exe` path) — public whenever the stack runs.
- **STUN-only** (`WEBRTC_PUBLIC=1`, no TURN). TURN is a ~10-min reactive add-on later if a specific
  visitor behind symmetric NAT can't connect.

## Design

### 1. `.env`
Set `WEBRTC_PUBLIC=1` (was `0`). Leave `TURN_URLS` empty (STUN-only). `WEBRTC_ICE_SUBNET` stays
`100.64.0.0/10` — in public mode `_restrict_ice_to_subnet` already keeps the Tailscale set **plus**
the default-route `/32` (the public-srflx interface), so tailnet + same-LAN + external all work.

### 2. `scripts/tunnel.ps1` (new, reusable)
Starts a Cloudflare quick tunnel to a local port, waits for the public URL, prints the shareable
link, and either holds the window (standalone) or returns the process+URL (for the launcher).

- `param([int]$Port = 7860, [switch]$Wait)`
- `Start-Process` the cloudflared exe with `tunnel --no-autoupdate --url http://localhost:$Port`,
  stdout+stderr redirected to `logs\cloudflared.log`, `-PassThru`.
- Poll `logs\cloudflared.log` (~30s budget) for regex `https://[-a-z0-9]+\.trycloudflare\.com`.
- Print the base URL + the recommended client path (`/studio/` when `AVATAR_PRESET=leo`, else
  `/nimbus/`; note `/client/` as fallback).
- `-Wait`: `Read-Host` to hold the window; on exit `Stop-Process` the tunnel. Otherwise return
  `[pscustomobject]@{ Process=<proc>; Url=<url> }` for `launch.ps1` to track + kill.
- ASCII-only source, `#requires -Version 5.1`, resolve the cloudflared path (PATH or the known
  Program Files location), match the existing `launch.ps1` helper style.

### 3. `launch.ps1` (edit)
- After the pipeline + config panel are up (before opening the browser), call
  `& scripts\tunnel.ps1 -Port 7860`, capture `$tunnel = @{Process,Url}`.
- Add the public URL to the "RUNNING" banner (base + recommended path). If the URL didn't parse in
  time, print a warning pointing at `logs\cloudflared.log` (don't fail the launch).
- On shutdown, `Stop-Process -Id $tunnel.Process.Id` alongside the other services.
- cloudflared does not listen on a local port, so it is killed by PID (not `Stop-Port`).

## Non-goals / accepted tradeoffs (unchanged)

- **Single-client** (one talker at a time; shared-GPU avatar server) and **unauthenticated** (anyone
  with the link drives the GPU + cloud LLM/STT spend). User chose "no password". A gate (Cloudflare
  Access / a token) is a future option, not in scope.
- Quick-tunnel URLs are **random and change on each tunnel restart** — inherent to zero-setup quick
  tunnels; a named tunnel (stable custom domain) is the upgrade path if that becomes annoying.
- No TURN (symmetric-NAT visitors may fail to connect until TURN is added).
- Config panel (`:7870`) is **not** exposed — the tunnel points only at `:7860`.

## Verification

- `python -m scripts.preflight` clean (no import drift).
- Pipeline log shows `Public WebRTC ICE servers ENABLED ... STUN only` after restart with
  `WEBRTC_PUBLIC=1`.
- `logs\cloudflared.log` yields a `trycloudflare.com` URL; that URL + `/studio/` loads the page over
  HTTPS.
- Final leg (a real off-tailnet browser forming media) is the user's live test — open the link on a
  phone with wifi off / cellular. Server-side reachability is already proven (2026-07-09).
