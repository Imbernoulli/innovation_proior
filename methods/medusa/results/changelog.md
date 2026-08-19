# medusa changelog

## 2026-08-18 — epistemic correction (svfix pass)
- `results/reasoning.md` (rejection-sampling acceptance-rule paragraph): the prior svfix pass had
  added a sentence where the narrator claims to have actually wired rejection sampling in as the
  acceptance rule over the candidate tree, run it at real non-zero temperature, and observed the
  outcome ("the whole system comes out slower than plain greedy decoding... every bit of the
  parallel-verification speedup gets eaten by rejections"). Grounded in Together AI's own published
  ablation, but that ablation is this method's own result (Medusa's tree-based acceptance rule), not
  a prior-work fact that pre-dates the method — so under the epistemic rule (a single-turn proposal
  has no results of its own yet) it still had to come out. Rewrote as a prediction/risk the narrator
  reasons through rather than a completed run: kept the experiment design (wire rejection sampling in
  as the tree's acceptance rule, at real non-zero sampling temperature), kept the prediction (rollback
  tax could plausibly erase the whole parallel-verification speedup, landing at or below plain greedy),
  and kept the decision consideration that follows (need accept/reject to get cheaper as the tree
  grows, not just correct) that motivates the pivot to typical acceptance. No other passage in this
  file's svfix diff repeats the violation; `answer.md`/`train_answer.md` were not touched by the
  svfix pass for this method, so out of scope.
- The pivot away from strict rejection sampling toward typical acceptance does not depend on the
  removed claim for its justification — it already rests on the kept on-page toy computation
  (expected acceptance 0.9 < 1 under real sampling) and the general structural-tax argument, both
  untouched. So the landing remains justified without the removed observation; not flagged for the
  trajectory-conversion queue.
