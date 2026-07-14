"""VAD + turn-taking parameters (Silero, local, always-on).

stop_secs is the responsiveness knob:
- shorter  -> faster end-of-speech, but risks cutting the user off at a mid-sentence pause
- longer   -> safer, but the user waits longer before the bot even starts

SCOPE, so the knob isn't over-trusted: under the baseline (ALLOW_INTERRUPTIONS=1) main.py
passes NO user_turn_strategies, so pipecat's defaults apply and end-of-TURN is called by
TurnAnalyzerUserTurnStopStrategy (Smart Turn v3, semantic) -- the VAD supplies the speech
segmentation that analyzer runs on. So these params SHAPE turn-taking; they don't dictate it.

None of it shows up in TTFO: that stopwatch starts at t0 = "user stopped speaking", i.e. once
the turn has already been called. It is latency the user feels but the metric never sees --
tune it by ear, not by the TTFO number.

All four are .env-driven (config.py) so the web config panel can tune them without a code edit.
"""
from __future__ import annotations

from loguru import logger

from pipeline.config import config


def build_vad_params():
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams

    # Log the live values: these are panel-editable, and a knob you can't confirm took effect
    # is worse than no knob (the panel's Restart is what applies them -- a .env edit alone won't).
    logger.info(f"VAD: stop={config.vad_stop_secs:g}s start={config.vad_start_secs:g}s "
                f"confidence={config.vad_confidence:g} min_volume={config.vad_min_volume:g}")
    return SileroVADAnalyzer(
        params=VADParams(
            stop_secs=config.vad_stop_secs,      # silence needed to call end-of-turn
            start_secs=config.vad_start_secs,    # speech needed to call start-of-turn
            confidence=config.vad_confidence,    # model score above which a frame counts as speech
            min_volume=config.vad_min_volume,    # volume gate (raise on a noisy mic)
        )
    )
