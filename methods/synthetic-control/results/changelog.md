# synthetic-control changelog

## 2026-08-18 — epistemic correction (svfix regression-fails-from-extrapolation step)
- `results/reasoning.md` (regression-fails-from-extrapolation paragraph pair, ¶4–5): an earlier
  svfix pass (`0409ce9a3`) had replaced the narrator's on-page algebra (three donors at 2/5/9, a
  treated unit at 12, weight `12/9 = 1.333` unreachable by any real mixture) with a claimed
  first-person regression run — "I run it and check... pre-RMSE goes to zero... post-RMSE comes out
  at 0.54... worse than the 0.33... worse again than the 0.44" — reporting the method's own
  pre/post-RMSE numbers as an observed outcome. At this point in the frame the method has not run
  any experiment yet; this violates proposal voice regardless of whether the numbers were fabricated
  or, as the svfix commit message states, lifted from a real published source (Abadie &
  Vives-i-Bastida 2022 Fig. 7) — citing that source here would itself be an anachronism, since the
  method under construction *is* synthetic control and the cited paper postdates it.
  Reverted both paragraphs to the pre-svfix text: the hand-worked convex-hull counterexample
  (2, 5, 9 donors; treated value 12; unreachable weight 1.333; irreducible pre-fit gap of 3) that
  motivates the nonnegative-sum-to-one restriction. This is on-page computation (allowed), needs no
  external observation, and is fully self-contained. It also repairs a coherence break the svfix
  edit had introduced: a later, unchanged sentence — "the way the value 12 outside `[2, 9]` left a
  gap of 3 that no choice of convex weights could close" — referenced the worked example that the
  svfix edit had deleted, leaving a dangling reference in the intervening HEAD state.
- `results/answer.md`, `results/train_answer.md`: unchanged by the flagged svfix commit and contain
  no RMSE/regression-run claims; no violation found, nothing touched.
- Landing (convex-restricted weights as the counterfactual mechanism) remains fully justified by the
  restored algebra; no dependency on the removed observation, so this does not enter the
  trajectory-conversion queue.
