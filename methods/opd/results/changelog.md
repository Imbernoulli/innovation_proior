# opd changelog

## 2026-08-18 — epistemic correction (svfix pass)
- `results/reasoning.md` (dropping the Long term / γ=0 discount step): the prior svfix pass had
  turned this step from a pure algebraic surrogate argument into the narrator claiming to have
  actually run the γ>0 vs γ=0 matched ablation and reporting its observed outcome ("Nonzero
  discount doesn't win"), plus claiming to have actually plotted the per-token penalty map and
  reporting what it showed ("I plot the per-token penalty ... and it isn't flat: it spikes hard at
  the tokens that open a phrase ... Both checks point the same way"). Per the epistemic rule (a
  single-turn proposal has no results of its own yet, only trajectory-track observations do),
  removed the claimed observations. Kept both discriminating checks' design in full: (1) matched
  training runs at a few γ>0 values against γ=0, same teacher/rollouts/budget, compared on
  downstream accuracy; (2) plotting the per-token reverse-KL penalty across a teacher-correct
  completion to see whether it already spikes at the forking tokens that derail the derivation
  rather than only near the final wrong number. Kept each check's prediction (what a γ>0 win vs.
  no-win implies; what a spike-at-forks vs. spike-only-at-the-end penalty map implies) and the
  decision rule ("ship γ=0 only if both checks agree ... otherwise keep some γ>0"). Also softened
  the immediately following sentence ("With both checks in hand, the moment the Long term is
  gone...") to a conditional ("If both checks agree and the Long term goes...") since it had been
  quietly presupposing that both checks were run and passed. No other passage in this file's svfix
  diff repeats the violation; `answer.md`/`train_answer.md` were not touched by the svfix pass for
  this method, so out of scope.
- The γ=0 landing now rests on the algebraic surrogate argument (Single term = exact reverse-KL
  gradient; Long term's only justification is sparse reward, which this setting doesn't have) plus
  an explicit, checkable decision rule — not on the removed empirical claim. Whether γ=0 actually
  wins the ablation and the penalty map actually localizes the forks is an open question until a
  real trajectory run confirms or refutes the prediction. Flagged for the trajectory-conversion
  queue.
