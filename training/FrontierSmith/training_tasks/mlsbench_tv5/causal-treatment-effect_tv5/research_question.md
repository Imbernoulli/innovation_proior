Three datasets, three different currencies. Effects in the IHDP-like and
ACIC-like settings live on a unit scale, while the jobs-like setting
deals in economic outcomes hundreds of times larger — and its two error
metrics pass through a logarithm before entering the task score, so what
counts there is relative, not absolute, accuracy. Any piece of an
estimator that carries units of the outcome — a ridge penalty applied to
unscaled targets, a tree's minimum-improvement threshold, a clipping
constant, a convergence tolerance — is silently calibrated to one
currency and miscalibrated for the others. Methods "tuned on IHDP" fail
to travel for exactly this reason.

Deliver a scale-equivariant estimation pipeline: if every outcome were
multiplied by a thousand, the predicted effects should emerge multiplied
by a thousand, with the estimator's internal behaviour otherwise
unchanged. That property must be engineered — standardize outcomes and
covariates internally with statistics estimated on the training fold,
perform all fitting in the standardized space where hyperparameters are
dimensionless, and back-transform predictions exactly once at the exit.
Robust centering and scaling deserve consideration over moments,
since heavy-tailed economic outcomes can make a standard deviation
itself an unstable yardstick. Branching on sample size, dimension, or
any fingerprint of dataset identity falls outside the rules — one
dimensionless configuration has to carry all three currencies on its
own.

Both metrics on all three settings remain the judge, and the log scoring
of the jobs-like setting is the sharpest test: it rewards cutting error
by a factor, which only happens when the estimator is genuinely working
in that setting's own units rather than importing constants from
elsewhere. Demonstrate equivariance directly (rescale Y in a self-test
and show predictions track) and show no setting's accuracy was purchased
with another's currency.
