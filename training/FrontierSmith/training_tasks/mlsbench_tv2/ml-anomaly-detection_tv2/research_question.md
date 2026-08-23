The 60/40 split hands your detector a training sample that is anything but
clean: it inherits each dataset's full anomaly fraction, from one-in-forty
records on thyroid to nearly one-in-three on satellite. Whatever structure you
estimate from that sample — densities, neighbourhoods, covariances, isolation
depths — is partly an estimate of the anomalies themselves. Contaminated
neighbourhoods make a test anomaly look ordinary because its polluted
look-alikes sit inside the reference set; a thickened density ridge grows
around every anomalous cluster; and the masking is strongest exactly where the
contamination is heaviest.

Treat the fit set as untrusted. The contribution this variant asks for is the
recovery mechanism: how to separate a credible clean core from a polluted,
unlabeled sample before (or while) modelling normality — hard trimming, soft
down-weighting, robust estimators, or an iterative loop that alternates
scoring with refitting until the core stabilises. Two constraints define the
variant. First, no per-dataset tuning: the true contamination is never given,
so how much to distrust the sample must itself be inferred from it, or made
irrelevant by the design. Second, the recovery must not backfire in the benign
regime — discarding a third of thyroid, which is 97.5% clean, throws away
legitimate tail behaviour and damages both reported numbers.

The claim to defend on the fixed AUROC and F1 protocol: scoring against a
recovered core beats scoring against the raw polluted sample, with the margin
growing as contamination rises and no measurable loss where contamination is
mild. The scaffold performs one hard trim at a guessed rate and refits once;
making the distrust level data-driven and the recovery iterative is where the
work lies.
