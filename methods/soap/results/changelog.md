# Changelog

## 2026-08-18 — svfix(epistemic)
- The 2026-08-17 svfix pass (D_candidate, ICLR 2025 OpenReview rebuttal forum ID
  xZhXrpNf) grounded the eigenvalue-vs-basis-staleness diagnostic behind the SOAP
  design step, but wrote it into reasoning.md in first-person present tense as the
  narrator's OWN executed ablation ("run the cheap factorized version ... and check
  it against full Shampoo ... The factorized, every-step-eigenvalue version tracks
  the fully up-to-date preconditioner across refresh frequencies; Shampoo itself
  only catches up to it once its own refresh frequency drops to 1") -- a
  single-turn proposal has no results yet, sourced or not. Own-method ablation
  results belong only in trajectory observation turns.
- Rewrote the passage in results/reasoning.md to keep the hypothesis (it is the
  eigenvalues going stale, not the basis), the discriminating-experiment DESIGN
  (factorized/Adafactor version in the same eigenbasis, refreshed on Shampoo's
  exact schedule, both swept across the same range of refresh frequencies f,
  everything else matched), the PREDICTION (the factorized version should track a
  fully up-to-date preconditioner at any f; Shampoo should only close the gap at
  f=1), and the decision rule ("that is the test that decides it: if the gap only
  closes at f=1 ... the eigenvalues were the frozen piece"). Removed the claimed
  observed outcome.
- answer.md / train_answer.md were not touched by the 2026-08-17 svfix pass for
  this method, so no corresponding change was needed there.
- The landing (freeze the Shampoo eigenbasis for f steps, run a full diagonal
  Adam-style second moment in that basis updated every step) is unchanged and is
  now motivated by the prediction/decision-rule framing rather than an in-frame
  observation -- expected per the epistemic-fix rule; this unit needs conversion
  to a trajectory observation turn to supply the actual ablation result.
