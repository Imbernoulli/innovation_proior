# squeezenet changelog

## 2026-08-18 — epistemic correction (svfix observation removal)
- `results/reasoning.md` (delay-downsampling paragraph): an earlier svfix pass grounded the
  "touches zero parameters" claim by having the narrator report a concrete compute-cost check on
  "the design I already have the shape of" — total inference computation "around 3.8x higher (2.3TF
  to 8.7TF in the case I worked)" and "still the same ~30MB of weights." Those numbers don't derive
  from anything on the page (no shown FLOP arithmetic), don't match this architecture's own
  later-computed size (~1.25M weight parameters, nowhere near 30MB), and read as an own-design
  measurement a single-turn proposal hasn't earned — a self-supplied observation dressed as a check.
  Rewrote to keep the two genuine algebra points the svfix pass surfaced: (1) a conv layer's weight
  count is a function of channel counts and kernel size only, so height/width truly never enter it
  (params-untouched claim proven, not asserted); (2) FLOPs do scale with height×width, so pushing a
  stride-2 step later doubles the map for every downstream layer and roughly quadruples each of
  their per-layer costs — a direct algebraic consequence, not a run result. Removed the fabricated
  aggregate multiplier/TFLOP/MB figures and replaced with an explicit "can't put an exact multiplier
  on that without profiling the built network" plus the decision rule already present in the original
  text (accept the FLOPs cost because parameters, not FLOPs, is the fixed metric). `answer.md` /
  `train_answer.md` had no svfix diff for this method, so no violation there. Landing (delay
  downsampling into the macro-architecture) unaffected — it was already presented as the chosen
  trade-off, not as a reported outcome.
