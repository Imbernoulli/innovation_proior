Recall is the metric that starves first on the Erdos-Renyi settings of this benchmark. At
fifty nodes the true graph carries hundreds of edges; a screening rule built on marginal
correlation hands most of its budget to ancestor pairs and hub shadows — associations that
are real but indirect — while the direct edges that F1 actually credits go unreported. The
usual repair, loosening the threshold, buys recall with a precision collapse that the
aggregate score punishes immediately. This variant asks for recall obtained the harder way:
by making the screening stage distinguish direct from indirect association before any
orientation is attempted.

The operative constraint is that abstention is not available as a safety valve. Every pair
the screen admits must be oriented and reported — discarding an admitted edge because its
direction looks uncertain is against the rules here — so false-positive control has to live
inside the admission decision itself, through conditional rather than marginal evidence:
partial correlation structure, sparse inverse-covariance support, residual-based screens, or
any mechanism that asks whether an association survives accounting for the remaining
variables. One configuration faces a thirty-node graph, a fifty-node graph with twice the
samples, and a hundred-node scale-free graph, and the same admission rule must neither flood
the dense settings nor starve the sparse one.

What has to be shown is a simultaneous move: directed-edge recall on the dense regimes
rising materially over correlation-screened baselines while precision holds its ground, and
SHD falling because the newly admitted edges are predominantly real. Recall bought by
threshold inflation, with precision paying the bill, is the outcome this variant is designed
to rule out.
