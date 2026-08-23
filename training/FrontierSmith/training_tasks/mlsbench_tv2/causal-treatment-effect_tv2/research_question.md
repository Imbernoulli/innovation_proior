PEHE is a root-mean-square over individuals, and its mass is not spread
evenly: for any estimator that leans toward the average effect, the
squared error concentrates on the people whose true effect sits far from
that average — the tails of the heterogeneity distribution. A predictor
can post a flattering ATE error while systematically failing exactly the
individuals for whom treatment-effect estimation matters, and the bulk of
its PEHE comes from that failure. All three synthetic settings here
contain such structure: nonlinear effect surfaces in the small-sample
setting, economically meaningful effect dispersion in the medium one, and
high-dimensional effect modifiers in the large confounded one.

The mandate of this variant is deviation-first estimation. Model the
centered effect field tau(x) minus its mean as the primary object:
identify which covariates modulate the effect, allocate model capacity
to the regions where the modulation is strongest, and resist the
shrinkage-to-the-mean reflex that minimizes variance at the cost of
erasing the signal PEHE actually measures. Techniques are open —
pseudo-outcome regressions weighted toward large deviations, effect
modeling with capacity concentrated on detected modifiers, two-stage
schemes that first locate heterogeneity then estimate it — but flat or
near-flat prediction is defined as failure here even where it would
score tolerably.

Success must be argued from both reported metrics jointly across the
three settings and ten replications: PEHE falling because tail errors
fell (inspect where the squared-error mass sits before and after), and
ATE error kept in check as a constraint rather than pursued as the
prize. An estimator that wins ATE error while its PEHE reveals
mean-collapse has answered a different question than this one.
