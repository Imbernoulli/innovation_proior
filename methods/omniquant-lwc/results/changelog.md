# omniquant-lwc changelog

## 2026-08-18 — epistemic correction (svfix)
- `results/reasoning.md` (the paragraph after the fifteen-weight toy-group sweep, before the
  transition into learning the clip): the prior svfix pass (`379d14100`, grounding the
  learn-vs-set clip pivot in a controlled ablation) had the narrator claim to have actually run
  the block-wise reconstruction pipeline on LLaMA-1-7B at W4A4, comparing a grid-searched
  per-group clip against a gradient-learned one under matched budget, and report the outcome as
  an observed result ("the grid-searched clip lands at 15.82 average perplexity; gradient descent
  ... lands at 12.87 ... still loses to a learned one"). Real numbers or not, at this point in the
  frame the method's own experiments have not happened yet, so a narrator reporting an observed
  outcome of its own ablation is out of voice for a single-turn proposal.
- Fix: removed the claimed run and both reported perplexity numbers (15.82, 12.87), and the
  outcome-dependent conclusion ("still loses to a learned one") stated as settled fact. Kept
  everything else the prior svfix pass added: the discriminating-experiment design (real
  block-wise pipeline, identical reconstruction objective, matched calibration budget, grid
  search over per-group candidate ratios vs. gradient descent on the same per-group ratio), the
  hypothesis for why a learned clip should win (a discrete sweep can only land near the optimum
  of a jagged group-specific loss surface; a gradient can reach wherever that surface's minimum
  actually sits), the explicit prediction (grid search will not match the learned clip), and the
  decision rule (whichever ratio-finding method reaches the lower loss under the identical budget
  is the one shipped; if grid search wins or ties, learning the clip is not built).
- The landing — committing to build the learned per-group clip as the method — is now stated as
  a bet ("betting on the gradient is what points away from setting the clip... pending that
  comparison") rather than a settled conclusion, since the observation that justified it is gone.
  This is expected per the epistemic-fix rule: the decision rule carries the proposal's honesty.
  Flagging this unit for the trajectory-conversion queue to supply the real ablation as an
  observation turn.
- No changes needed to `answer.md` or `train_answer.md`: their svfix diffs (none — this method's
  svfix commit only touched `reasoning.md`) contain no claimed run or reported numbers.
