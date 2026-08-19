# Changelog

- 2026-08-18 `results/reasoning.md` — epistemic fix (svfix epistemic pass).
  The prior svfix commit (`3f1552880`) had closed the Chinchilla->inference-optimal
  reframe's load-bearing premise by having the narrator claim to run the actual
  7B pretraining ("I run the actual pretraining and watch loss against tokens
  through the run... It doesn't flatten... the training loss is still
  descending at 1T... confirmed on the curve rather than assumed") — a
  self-supplied own-method observation a single-turn proposal is not entitled
  to state. Rewrote the passage to remove the claimed run and its outcome
  while keeping: the hypothesis (loss may or may not have saturated by 1T
  tokens), the discriminating-experiment design (push the 7B's run past the
  Chinchilla-assigned stopping point and watch loss vs. tokens), the
  prediction ("my expectation... is that a 7B is nowhere near saturated...
  but I'm flagging it as expectation, not established fact"), and the
  decision rule (descending-on-slope => headroom real, trade holds;
  bent-flat => small model never reaches L* and the plan collapses). Kept
  the Chinchilla-own-rule fact (10B -> 200B tokens) the svfix pass had added,
  since that is a prior-work published fact, not an own-method observation.
  Landing (build the inference-optimal family, 7B-65B on ~1-1.4T tokens)
  is now carried only by the decision rule, not a settled result — flagged
  for trajectory-conversion so the "it holds" branch gets a real observation
  turn. No changes needed in answer.md/train_answer.md (svfix diff for this
  method touched only reasoning.md).
