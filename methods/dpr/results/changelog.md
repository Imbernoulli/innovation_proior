# dpr changelog

## 2026-08-18 — epistemic correction (svfix)
- `results/reasoning.md` (training-objective section, pointwise-BCE draft): the prior svfix pass
  (`0a06b9b14`) had the narrator claim to have actually trained the rejected pointwise-BCE
  objective and observed poor retrieval numbers ("when I actually train with it the retrieval
  numbers come out poor — worse than the cleanliness of the objective would suggest"). At this
  point in the frame the method's own experiments have not happened yet, so a stated observation
  — even a qualitative one, with no numbers — is out of scope for a single-turn proposal.
- Fix: removed the claimed training run and its outcome. Kept everything else the prior svfix pass
  added: the pointwise-BCE draft objective itself, the on-page gradient derivation
  (∂L/∂sim(q,p) = P(q,p) − y, no cross-candidate term), the argument that a per-pair global
  threshold doesn't encode the actual serving-time ranking condition, and the shift-invariance
  argument for why the softmax alternative fixes this. The theoretical/gradient argument alone
  fully justifies rejecting the pointwise draft and moving to the softmax objective, so the landing
  is unaffected — no trajectory-conversion needed for this passage.
