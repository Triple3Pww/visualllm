# IEEE architecture paper (working draft)

- `draft.md` — the paper (Markdown; converts to IEEEtran LaTeX after the prof approves content).
- `references.bib` — BibTeX; draft cites with pandoc `[@key]`.
- `data/` — frozen measurement runs (`{zh,en}_run_NN.json` = copies of `output/measure_report.json`),
  `environment.md` (commit + .env snapshot the numbers were taken on), `summary.md` (generated).
- `scripts/` — `aggregate.py` (data -> summary.md), `make_waterfall.py` (data -> figures/waterfall.*).
- `figures/` — generated + hand-authored SVG figures.

Spec: `docs/superpowers/specs/2026-07-15-ieee-architecture-paper-design.md`.
Do not push without the user's go-ahead (public repo).
