# timemixer changelog

## 2026-08-18 — epistemic correction (svfix follow-up)
- `results/reasoning.md` (season/trend routing paragraph): the prior svfix pass (`W3_primary_plus_ancestors`)
  correctly restaged the season-up/trend-down routing from an asserted first-principles claim into a
  hypothesis with a 4-way discriminating experiment (both bottom-up, both top-down, my guess, full
  reverse; decomposition/pooling/mixer width/layers held fixed), but it also had the narrator report
  the outcome — actual MSE/MAE numbers from a direction-sweep on ETTh1-336 — inside `reasoning.md`.
  Per the proposal-voice rule, a single-turn method unit's own experiments have not happened yet in
  frame, so the narrator cannot claim to have run the sweep and read off results. Removed the
  observed numbers and the conclusion drawn from them; kept the hypothesis, the experiment design
  (controls, matched budget, benchmark windows), a falsifiable prediction (wrong pairing costs more
  than either single flip alone), and the decision rule ("whichever routing comes out ahead is the
  one I ship"). The implementation that follows is now framed as building the predicted winner so the
  design is ready to test, not as a routing already validated by observed numbers.
- The landing (season-up/trend-down as the shipped routing) is no longer justified by an in-frame
  observation — expected, since the observation was the violation. This unit needs a trajectory
  observation turn to carry the actual sweep result.
