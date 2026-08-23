Repetition is the hidden examiner in this benchmark. The harness refits
the estimator from scratch on every one of five folds in every one of ten
replications per dataset, and the reported PEHE and ATE error are
averages over all of those refits. An estimator whose predictions swing
with the training split pays for that instability directly: variance
across refits inflates a mean-squared metric even when the average
prediction surface is right. Nowhere does this bite harder than in the
747-row, 25-covariate nonlinear setting, where flexible learners see a
different world in every fold and heavily regularized ones flatten the
effect surface toward zero.

The assignment is explicit bias-variance governance for small-sample
CATE. Make stability a designed property: aggregate over internal
resampled refits so single-split accidents average out, choose shrinkage
strength by a data-driven rule instead of a fixed constant, and expose a
diagnostic for how much your predictions disperse across internal refits
so the stability claim is measurable rather than asserted. The trap on
each side is named: chasing stability into a constant prediction throws
away the heterogeneity that PEHE prices, while refusing regularization
lets fold-to-fold variance eat the metric. The interesting region is
between, and finding it from the data — not from knowing which dataset
is loaded — is the contribution.

The 2000- and 4000-row settings act as the control group: a method that
buys small-sample stability by crippling its capacity will be exposed
there, since those samples can support richer fits. The winning profile
is uniform: improved PEHE in the small setting driven by variance
reduction, no regression on the larger two, and ATE error steady
throughout — all from a single configuration across every fold,
replication, and dataset.
