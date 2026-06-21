## Problem

The probabilistic method often proves existence by choosing a random object, defining bad events
`A_1, ..., A_n`, and showing that there is positive probability that none of them occurs. In symbols,
the target is `P(no A_i occurs) > 0`. A good outcome is then an object satisfying every desired local
constraint.

The difficult regime is not when every bad event is rare in isolation. It is when there are so many
bad events that their probabilities add up to much more than one, while the events are also not
globally independent. This is the gap Lovasz Local Lemma fills: many individually unlikely failures,
not mutually independent, but only locally entangled.

## Earlier Baselines

Full independence gives an easy product:
`P(no A_i occurs) = prod_i (1 - P(A_i))`, which is positive when each `P(A_i) < 1`. But in most
combinatorial constructions, two local failures can share random variables, so mutual independence of
all bad events is too strong.

The union bound gives the other easy tool:
`P(some A_i occurs) <= sum_i P(A_i)`. It requires no independence, but it is globally conservative.
Every bad event is charged at full price, including events that are disjoint from the one currently
being considered and hence should not matter. Once `sum_i P(A_i) >= 1`, the union bound cannot prove
that a good object exists, even if each bad event interacts with only a few others.

## Local Dependency Structure

The key structure is a dependency graph whose vertices are the bad events. An edge means two events
may depend on each other. The required condition is not merely pairwise independence: each `A_i` must
be independent of the whole set of its non-neighbours. This set-level independence is what lets the
proof ignore all far-away bad events at once.

In common applications this graph is sparse because each bad event depends on only a small set of
underlying random choices. In hypergraph two-colouring, the bad event for an edge depends only on the
coin flips at that edge's vertices, so it is independent of bad events for disjoint edges. In `k`-SAT,
a clause-violation event depends only on clauses sharing variables with it.

## Core Reframing

Lovasz Local Lemma replaces the global question "is the sum of all bad-event probabilities less than
one?" with the local question "is each bad event sufficiently unlikely compared with the number and
weight of the bad events it can depend on?" The proof uses the chain rule:
`P(no A_i occurs) = prod_i P(A_i does not occur | earlier bad events did not occur)`.

Thus the right quantity is not the unconditional `P(A_i)` used by the union bound. It is the
conditional probability that `A_i` happens after some other bad events have been avoided. Splitting
the conditioning into neighbours and non-neighbours is the essential move: non-neighbours disappear
by independence, while neighbours cost only a local product. In the asymmetric form, if there are
numbers `x_i in [0,1)` such that
`P(A_i) <= x_i prod_{j in N(i)} (1 - x_j)`, then
`P(no A_i occurs) >= prod_i (1 - x_i) > 0`.

## Consequences

The symmetric corollary says that if every bad event has probability at most `p`, each event depends
on at most `d` others, and `e p (d + 1) <= 1`, then all bad events can be avoided simultaneously with
positive probability. This is the distinctive insight: bad events need not be globally independent;
local dependence is acceptable when the dependency graph is sparse and each event is unlikely.

For a `k`-uniform hypergraph, a random two-colouring makes a fixed edge monochromatic with probability
`2^{-(k-1)}`. The union bound controls only the total number of edges. The local lemma instead controls
how many other edges a given edge intersects. For `k`-SAT, a violated clause has probability `2^{-k}`;
the relevant quantity is how many clauses share variables with it, not how many clauses exist in
total.
