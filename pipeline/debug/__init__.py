"""Visual debug dashboard for the pipeline.

A bolt-on observability layer — it does NOT change the frame path. `StageTap`
processors (pass-through) report per-stage activity to a single in-process
`StatusBus`; a small FastAPI server on its own port streams that bus to a browser
dashboard so you can see, at a glance, which stage is working and which is broken.

Everything here is additive and gated behind `config.debug_dashboard` (DEBUG_DASHBOARD).
"""
from pipeline.debug.status_bus import bus

__all__ = ["bus"]
