# Changelog

## 2026-08-18/19 — svfix(W3_ancestors_only)

- Verified the decisive step (kill-the-oldest replacing kill-the-worst
  eviction) already runs through the real forcing argument, verbatim-matched
  to `src/real2018_regularized_evolution.tex` Discussion section (line 370).
  No sourcing gap there; left unchanged.
- Fixed a self-supplied-observation defect in `results/reasoning.md`: the toy
  unit-cube verification previously had the think-voice claim to "run" the
  simulation and report a fabricated 5-row numeric table (D=10..500,
  kill-oldest vs kill-worst denoised-fitness values). No such table exists in
  the primary source — Supplement C reports only a figure with a qualitative
  claim ("AE is never worse and is significantly better for larger D", line
  658). Rewrote the block to hypothesis -> test design (matched budgets,
  denoised-fitness metric to avoid the circular raw-accuracy trap) ->
  falsifiable prediction (tie at low D, aging ahead as per-bit signal
  approaches the noise floor) -> decision rule, and correspondingly softened
  the closing recap's reference to the toy experiment so it no longer asserts
  an observed outcome. Landing (kill-the-oldest method + code) unchanged.
  See `notes/sources.md` for the verbatim quotes backing both points.
