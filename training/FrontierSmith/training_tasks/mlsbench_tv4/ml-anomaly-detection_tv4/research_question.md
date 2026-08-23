"Anomaly" is not one population. In these benchmarks it covers marginal
outliers that stick out on a single coordinate, correlation breakers whose
every coordinate is individually unremarkable, small tight clusters of
repeated faults, and diffuse noise records — and the four datasets mix these
types in different, unknown proportions. Every classical detector family has
a blind spot in that taxonomy: per-dimension tail methods cannot see a broken
correlation, global covariance methods excuse anomalies hiding inside dense
local pockets, and neighbour-distance methods forgive a fault that arrives as
a cluster of mutually similar copies.

The premise of this variant is that your result on any dataset is capped by
the type your method is blind to, because per-dataset AUROC and F1 are
computed over all of its anomalies at once: missing an entire type concedes
both numbers wherever that type is common. So build for the worst-covered
type, not the average one. The contribution is two-fold: a basis of scorers
whose blind spots provably differ — marginal density against dependence
structure against locality — and an aggregation rule that keeps the ensemble
honest when the families disagree, rather than letting the majority view
average a minority type's evidence away.

Constraints: one fixed configuration across all four datasets, no labels, and
no assumption about which type dominates where. Defend this on the unchanged
metrics: against each single-family baseline, the combined detector loses
little where that family shines and wins large where that family is blind, so
its worst dataset sits well above any single family's worst. The scaffold
provides three deliberately complementary scorers joined by a plain mean of
standardised scores — an aggregation that still dilutes minority evidence,
and the first thing worth replacing.
