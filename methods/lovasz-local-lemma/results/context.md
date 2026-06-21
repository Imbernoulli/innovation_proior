## Problem

The probabilistic method proves existence by choosing a random object, defining bad events
`A_1, ..., A_n`, and showing that there is positive probability that none of them occurs. In symbols,
the target is `P(no A_i occurs) > 0`. A good outcome is then an object satisfying every desired local
constraint.

The regime of interest is when there are many bad events whose probabilities sum to much more than one,
while the events are not globally independent but only locally entangled with each other.

## Earlier Baselines

Full independence gives an easy product:
`P(no A_i occurs) = prod_i (1 - P(A_i))`, which is positive when each `P(A_i) < 1`. But in most
combinatorial constructions, two local failures can share random variables, so mutual independence of
all bad events is too strong.

The union bound gives the other easy tool:
`P(some A_i occurs) <= sum_i P(A_i)`. It requires no independence and charges every bad event at full
price regardless of which other events it interacts with.

## Local Dependency Structure

The key structure is a dependency graph whose vertices are the bad events. An edge means two events
may depend on each other. The required condition is not merely pairwise independence: each `A_i` must
be independent of the whole set of its non-neighbours. This set-level independence allows the analysis
to ignore all far-away bad events at once.

In common applications this graph is sparse because each bad event depends on only a small set of
underlying random choices. In hypergraph two-colouring, the bad event for an edge depends only on the
coin flips at that edge's vertices, so it is independent of bad events for disjoint edges. In `k`-SAT,
a clause-violation event depends only on clauses sharing variables with it.

## Research Question

Given a probability space with bad events `A_1, ..., A_n` that are not globally independent but have
sparse local dependencies, how can one certify that `P(no A_i occurs) > 0` beyond what the union bound
or full independence allows?
