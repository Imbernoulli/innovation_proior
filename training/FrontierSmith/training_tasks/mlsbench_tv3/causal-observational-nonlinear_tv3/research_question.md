Finding which pairs of variables are linked is the easy half of this
benchmark; deciding which way each link points is where scores are won or
lost. Because the metrics count an edge as correct only when skeleton and
arrowhead both agree with the truth, a method that recovers every adjacency
but orients coin-flip converts half of its correct detections into
simultaneous false positives and false negatives — a heavier penalty than
never having claimed those edges. The evaluation grid is built to stress the
orientation machinery specifically: Gaussian noise nearly erases the additive
residual asymmetry that identification relies on, mixed function families
break any orientation rule tuned to one smoother, and 150 samples of twelve
variables leave direction statistics with wide error bars.

The brief is therefore arrow-first engineering. Fix the adjacency
step to something unremarkable and pour the effort into the two-way contrast:
regress each endpoint on the other with a flexible one-dimensional smoother,
quantify how strongly each residual still depends on its regressor, and let
the arrow follow the direction whose residual is cleaner. The decisive design
choices are the residual-dependence functional itself (moment proxies,
kernel-based criteria, entropy scores), how its two directional values are
turned into a confidence margin, and what happens below the margin — commit,
abstain, or drop the edge outright. An orientation rule that keeps its
accuracy when the noise is Gaussian and the sample is small is the deliverable
here; the abstention policy determines whether weak evidence bleeds precision
or recall.

Every knob of the harness — generator grid, metric definitions, aggregation —
is inherited verbatim from the parent task. The defended claim is that a margin-aware
residual-asymmetry orienter lifts directed scores over both a coin-flip
orienter and an always-commit orienter on the identical skeleton, with the
Gaussian-noise setting — not the friendly exponential one — as the exhibit.
