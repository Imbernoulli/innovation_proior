When a fixed classifier must hand a fifth of its decisions to a human reviewer,
*who absorbs the deferrals* becomes the scientific question. A single
confidence cutoff concentrates abstention on whichever subgroup the base model
finds hardest, so one population quietly carries most of the review burden;
per-group cutoffs equalize that burden by construction, but they re-rank
examples inconsistently across groups, and an acceptance score that is no
longer globally comparable stops being a trustworthy correctness signal. This
variant targets the reconciliation: one score, group-equitable deferral, and no
collapse in worst-group reliability.

Concretely, at the fixed 80% budget the policy should drive the spread in
per-subgroup deferral rates toward zero, hold the error rate of the worst
subgroup among accepted cases rather than trading it away for burden parity,
and keep the score itself — judged as a ranking of correctness over the whole
test split — discriminative. All of these are measured on every dataset, and
the task score multiplies across datasets, so repairing the gap on one dataset
by wrecking another is self-defeating. Achieved coverage is tracked too:
landing materially off the budget forfeits whatever risk advantage the smaller
accepted set appeared to buy.

Fitting happens offline on calibration probabilities, labels, and subgroup
ids; group ids are visible again at prediction time. The interesting designs
use them to *reshape a single comparable score* — per-group corrections folded
into one scale, burden-aware offsets, calibration of the confidence signal
itself — rather than to operate disjoint per-group policies whose scores mean
different things. Group-blind designs are admissible but must then explain how
they protect a subgroup they never distinguish.

The claim to defend at the end is causal, not numerical: identify the
mechanism by which equalizing deferral burden avoided the usual payment in
worst-group accepted error or in global ranking quality, and show where a
naive equalizer would have paid it.
