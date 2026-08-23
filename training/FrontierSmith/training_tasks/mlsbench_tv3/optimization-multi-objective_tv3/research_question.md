Of the three numbers scored per problem, the evenness term is the most concretely computed: for
two objectives the final set is sorted along the first objective and the metric is the mean
absolute deviation of consecutive gaps, normalised by the mean gap; for three objectives it is
the dispersion of nearest-neighbour distances. Low means the front reads like a ruler — equal
steps, no thin stretches, no clumps. This variant elevates that term to the primary discipline:
the algorithm must close the largest hole in its own front, generation after generation,
without letting convergence pay for it.

Two places in the loop can enforce the discipline. Mating can be aimed: locate the widest gap in
the current non-dominated set and recombine exactly across it, so offspring land where coverage
is thinnest rather than where parents happen to be dense. Truncation can be principled: when the
last front must be cut, repeatedly deleting whichever member sits closest to its nearest
neighbour — with the objective-wise extremes protected — is a direct descent on the dispersion
the scorer computes, where crowding-distance ranking is only a loose surrogate of it.

The guardrails are the other two numbers. A front can be made perfectly even by shrinking its
reach, or by spacing itself along a surface that has stopped approaching the true trade-off;
both cheats are visible, one in dominated volume and the other in reference-front distance, and
both count as failures of the mechanism rather than clever accounting. Blindness still applies —
the strategy never learns which landscape it is on, must run identically for two and for three
objectives, and takes whatever population and generation budget the spec dictates.

Defend the result with a per-generation trace of the internal evenness statistic alongside the
final three-number readout: the trace should show holes closing early and staying closed, and
the readout should show nothing else was surrendered to achieve it.
