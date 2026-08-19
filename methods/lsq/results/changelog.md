# Changelog

- 2026-08-18 `results/reasoning.md` — epistemic fix (svfix pass on
  2026-08-18): the earlier svfix commit (`svfix(D_candidate)`) had the
  narrator *run* the hand-tuned-multiplier baseline (six 2-bit ResNet-18
  trainings, 9 epochs each, sweeping the LR multiplier over
  {10⁰,…,10⁻⁵}) and *report* its outcome ("best point at 10⁻⁴"), and
  separately had the narrator train a 2-bit ResNet-18 with no gradient-scale
  correction and report a convergence failure / −3.4 top-1 gap versus the
  corrected version. Both are the method's own not-yet-run experimental
  results stated in proposal voice, which the frame rule forbids (results
  belong only in trajectory observation turns). Rewrote both passages to
  keep the hand-tune baseline's experiment *design* (six-run, 9-epoch,
  six-point sweep) and the *reasoning* for why even a successful sweep
  wouldn't generalize per-layer, and to keep the severity concern about the
  imbalance as a stated worry ("I doubt this is just a mild inefficiency...
  looks large enough to threaten convergence outright") rather than a
  reported measurement. Landing (derive g = 1/√(N_W·Q_P) from the R
  Monte-Carlo, ship gradient-scale correction) unchanged and still
  justified by the decision rule already in the text; the pre-existing R
  Monte-Carlo table (on-page numerical simulation, not a training run) was
  untouched — out of scope and not a violation. answer.md/train_answer.md
  had no corresponding svfix diff, so left untouched this pass.
