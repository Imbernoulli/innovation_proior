# circle-single-slsqp changelog

## 2026-08-17 — source-value recheck
- `results/reasoning.md` (seed-spread paragraph): corrected a factual error. The text claimed the
  restart seeds `1, 2, 3, 7` (`2.581, 2.581, 2.611, 2.592`) had "none reaching seed `0`'s `2.595`
  from above", but seed `3`'s `2.611` clears `2.595`. Reworded to state that one seed does beat
  seed `0`; the conclusion drawn from the scatter (a single random basin is what caps the score) is
  unchanged and in fact strengthened. No other file repeats the claim; landing and code unchanged.
- Grounding verdict: this rung is a self-generated optimization run, not a reconstruction of an
  external paper, so there is no non-primary source to run the decisive step through. The decisive
  step (fixed centers ⇒ the radii subproblem is an LP) is verified in-trace by a hand computation on
  3 circles (walls `0.2/0.5/0.2`, `d₀₁=d₁₂=0.3`, so `r=(0.2,0.1,0.2)`, sum `0.5`) that the LP
  reproduces; feasibility (`1.7e-17`, `1.2e-14`), the null re-tightening gain (`-5.5e-14`) and the
  seed scatter are all reported honestly, including the negative results. Legitimate self-derivation.
