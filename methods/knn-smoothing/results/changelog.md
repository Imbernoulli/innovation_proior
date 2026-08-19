# knn-smoothing changelog

## 2026-08-18 — epistemic correction (svfix)
- `results/reasoning.md` (neighbors-on-raw-profiles section, and its callback in the
  variance-decomposition prediction paragraph): the prior svfix pass (`e57626690`, grounding the
  raw-profile neighbor-selection weakness in a documented size/depth-noise bias) had the narrator
  claim to have actually simulated 400 cells from one true state with lognormal capture-depth
  variation, run the exact neighbor-selection recipe, and read off the result ("mean depth of
  picked neighbor pool: 6430.1... fraction of picks from the deep half: 0.996... holds up across
  reruns, ratio 2.7-3.1 each time"). Real numbers or not, at this point in the frame the method's
  own experiments have not happened yet, so a narrator reporting an observed outcome of its own
  check is out of voice for a single-turn proposal.
- Fix: removed the claimed simulation run and every reported number (2213.9, 6430.1, 2.90, 0.996,
  the 2.7-3.1 reseed range). Kept everything else the prior svfix pass added: the mechanism/
  hypothesis (a deeply-captured cell keeps proportionally less per-gene Poisson noise than a
  shallow same-state cell, so normalizing does not remove the noise-level confound from neighbor
  search), the discriminating-check design (simulate matched-state cells with realistic
  depth variation, compare picked-neighbor depth/fraction against the population and the 0.5
  unbiased baseline, check stability across random seeds), and the explicit prediction (picked
  pool should skew well above 0.5 toward the low-noise half). Reworded the check's callback later
  in reasoning.md ("the size-bias check shows...") to the same hedged, not-yet-run register
  ("the size-bias argument says... I would expect to survive here").
- The decision to accept raw-profile neighbor search as deliberately crude, and to defer curing it
  to a graph-diffusion rung, does not depend on the removed observation — it was already the
  rung's stated character before this check existed, and the paragraph still lands on that same
  "accept on purpose, cure it next rung" choice with only the prediction (not a settled result)
  behind it. So no landing is left unjustified and no trajectory-conversion flag is needed.
- No changes to `answer.md` or `train_answer.md` were needed: their svfix diffs state the same
  mechanism as a design rationale (no claimed run, no reported numbers), so nothing there
  violates the rule.
