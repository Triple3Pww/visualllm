"""VAD + turn-taking parameters (Silero, local, always-on).

The silence threshold is the biggest single lever on TTFO vs responsiveness:
- shorter  -> faster turn-end detection, but risks cutting the user off
- longer   -> safer, but adds directly to TTFO
Start at 0.5s and tune in Phase 1/4.
"""
from __future__ import annotations


def build_vad_params():
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams

    return SileroVADAnalyzer(
        params=VADParams(
            stop_secs=0.5,      # silence needed to call end-of-turn
            start_secs=0.2,     # speech needed to call start-of-turn
            confidence=0.7,
            min_volume=0.6,
        )
    )
