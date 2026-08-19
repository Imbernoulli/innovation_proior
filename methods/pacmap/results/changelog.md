# Changelog — pacmap

## 2026-08-18 — svfix(epistemic)
- **Removed a self-supplied observation introduced by the svfix(B_selfaccount_candidate)
  pass** (commit `e0ee34c23`, "decisive local-vs-global-structure step now grounded in an
  actual built-and-run test"). A single-turn method unit is a proposal: the method's own
  experiments (and diagnostic experiments run mid-derivation against a baseline) have not
  happened at that point in the frame, so reasoning.md must not have the narrator run an
  experiment and report its result — real numbers or not. The prior pass replaced an
  abstract thought-experiment (a pure algebraic construction showing a 0-1 triplet loss
  *can* be driven to zero while scrambling global layout) with a claimed built-and-run test
  — "build a 2D curve with four turns ... and run UMAP on it ... That loss comes out
  essentially zero ... the layout itself is wrong ... and I got there by actually running
  it, not just positing it" — narrating UMAP's behavior on a synthetic curve as an observed
  outcome. That is the narrator running a diagnostic experiment mid-derivation and stating
  the result, which the frame does not allow, even though the subject is a prior-work
  baseline (UMAP) rather than PaCMAP itself and even though the underlying claim traces to
  real figures in the paper source (`src/Figures_final/2D_curve_line_datasetUMAP.png`) and
  the Rudin WNAR-2023 self-account.
- Rewrote the passage to keep the discriminating-experiment DESIGN (the specific 4-turn
  curve construction, chosen so along-curve closeness and embedded closeness can genuinely
  come apart; UMAP as the test subject because it already satisfies every local principle
  derived above; the 0-1 triplet scoring rule restricted to genuine near-neighbor triplets),
  the PREDICTION (the score should land at or near zero while turns of the curve that are
  far apart along the original path end up glued together in the projection), the
  supporting logical argument for why that would follow from the loss's blind spot to
  far-far relative arrangement (this is a deduction from the loss's definition, not an
  empirical claim, so it stays), and the decision rule (a folded layout under a
  near-perfect score is what confirms the worry; an unfolded layout would mean the worry
  was overblown) — without asserting the test was actually run or reporting its outcome.
- Landing (the conclusion that local-only losses need a third, non-neighbor force to carry
  global structure) is now argued from the predicted/designed test plus the deductive
  argument rather than from a reported observation — expected per the epistemic-fix rule;
  this unit needs conversion to a trajectory observation turn to supply the actual
  UMAP-on-curve result.
- No other svfix-diff passages in this method (answer.md/train_answer.md untouched by the
  B_selfaccount_candidate pass); scope was this one paragraph in reasoning.md.
