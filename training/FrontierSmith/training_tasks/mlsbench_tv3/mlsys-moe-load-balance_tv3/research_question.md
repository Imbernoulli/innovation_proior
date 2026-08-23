Four deployments, one algorithm, no knobs: that is the discipline of this
variant. The suite spans routing distributions from mildly uneven (Zipf
exponent one half) to pathologically long-tailed (exponent one), skew
mixtures from seventy to ninety-five percent, and topologies from four
nodes to sixteen — and the task score is a geometric mean, so a method that
shines on the profile it was tuned for and stumbles elsewhere is priced by
its stumble. The research question is whether a single untuned placement
rule can hold per-GPU balance, node balance, and locality simultaneously
across that whole spectrum, and across every layer of every trial within
it, since each layer draws its own workload and the harness averages over
all of them.

Forbidden by construction: per-configuration constants, thresholds chosen
by peeking at a profile name, branching on expert counts to pick a
strategy. Encouraged instead: decisions computed from the workload itself —
quantiles of the observed expert loads, mass-concentration statistics, an
effective count of hot experts — so that the same code path adapts because
its inputs change, not because its author anticipated the case.
Distribution shift is the actual test: a rule fitted to the head-heavy
regime must not fall apart when the tail thickens, and the reverse must
hold too.

Runtime and locality are scored alongside the balance terms, so adaptivity
cannot be implemented as an expensive per-layer solver, nor as replica
scattering that quietly trades interconnect traffic for flatness. The claim
to defend: across all four profiles the proposed rule's per-config scores
are uniformly strong — the spread between its best and worst configuration
is small — and no component of the method encodes which configuration it is
running on.
