# luffy changelog

## 2026-08-18 — epistemic correction (svfix pass)
- `results/reasoning.md` (mixed-objective "hacking" paragraph): the prior svfix pass had rewritten
  this step from a pure in-head derivation into the narrator claiming to have actually run the
  raw/unshaped off-policy ablation and reporting its observed training-curve outcome ("the model's
  own solve rate drops hard", curves "sit almost on top of each other", "still trails plain
  on-policy GRPO"). Per the epistemic rule (a single-turn proposal has no results of its own yet,
  only trajectory-track observations do), removed the claimed observations. Kept the discriminating
  check's design in full: raw/unshaped off-policy term vs. two matched controls (plain on-policy
  GRPO, plain SFT-on-teacher-trace), same checkpoint and step budget; kept the prediction (mixed run
  should hug the SFT curve and keep trailing GRPO if hacking is real, vs. tracking/beating GRPO if
  not); kept the decision rule ("whichever of those two shapes the curve actually takes is the test
  that decides"); kept the gradient-scaling mechanism argument (on-page algebra, unaffected by the
  correction) as the reason the predicted failure would occur if it does. No other passage in this
  file's svfix diff repeats the violation; `answer.md`/`train_answer.md` were not touched by the
  svfix pass for this method, so out of scope.
- The mixed-objective "hacking" step's landing (the `f(x) = x/(x+γ)` shaping fix) now rests on the
  on-page gradient derivation, not on the removed empirical claim; the ablation itself is an open
  question until a real trajectory run confirms or refutes the prediction. Flagged for the
  trajectory-conversion queue.
