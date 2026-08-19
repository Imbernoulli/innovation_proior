# Changelog

## 2026-08-18 — svfix(epistemic)
The svfix pass (commit 15dd45e2d) grounded the "SAB-before-PMA vs single-query
cross-attention" comparison in a real factorial ablation instead of arguing the two
additions (multihead pooling, self-attention encoder) bundled — but it wrote the
narrator as having actually *run* that ablation and read off results ("On a labeled
set-classification task I can check this against, a single content-dependent query
... barely clears the parameter-free mean-pooling floor — AUROC 0.5643 → 0.5671 ...
Swap that for self-attention ... the same floor moves ... to 0.5757 ... Stack both
... to 0.5941"). A single-turn proposal has no results yet, real numbers or not.

Rewrote the passage in `results/reasoning.md` to keep: the methodological move itself
(more parameters can win for boring reasons, so isolate the two additions instead of
arguing them bundled); the discriminating-experiment DESIGN (a four-condition factorial
on a labeled set-classification task, holding capacity comparable — mean-pool floor,
content-dependent query alone, SAB-interaction alone feeding an ordinary pool, and the
two stacked); each hypothesis's PREDICTION (content-dependence alone should barely move
off the floor if interaction is what matters; SAB-alone should clear the floor by more
than content-dependence alone if it is; the stacked condition should beat the sum of the
two isolated gains if the mechanisms are complementary rather than redundant); the
boundary condition as a prediction, not a settled finding (a large near-redundant set
could let plain pooling beat the content-dependent seed, since the seed's value should
be conditional on which-elements-matter still being in question); and the decision rule
(a jump no bigger than the sum of the isolated gains would say the two mechanisms are
redundant and one could be dropped). Removed the claimed run and all four reported AUROC
numbers (0.5643 mean-pool, 0.5671 dotprod-only, 0.5757 SAB-only, 0.5941 stacked).

The landing ("SAB + PMA" — keep both the encoder and the content-dependent seed) is not
solely carried by this passage: it is independently supported by the pre-existing,
untouched permutation-invariance construction check and the pre-existing max-value-
regression toy-task evidence earlier and later in the same section (both out of scope
for this pass). This specific factorial's outcome is still unknown at proposal time,
so this unit needs trajectory-track conversion to carry the real run and its result.
