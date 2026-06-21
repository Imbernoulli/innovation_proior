# Lovasz Local Lemma

The Lovasz Local Lemma's distinctive insight is that bad events do not have to be globally independent.
They may depend on nearby bad events, as long as the dependency graph is sparse and each bad event is
unlikely enough relative to that local neighbourhood.

The union bound proves `P(no bad event) > 0` only from a global budget:

```text
sum_i P(A_i) < 1.
```

That is often too conservative. It charges every bad event, including events that are disjoint from
one another and therefore statistically irrelevant to each other. The local lemma replaces this with
a dependency-graph condition. In its symmetric form:

```text
P(A_i) <= p for all i,
each A_i depends on at most d other events,
e p (d + 1) <= 1
        => P(no A_i occurs) > 0.
```

The proof works because the exact probability of avoiding all bad events is a product of conditional
terms:

```text
P(no A_i occurs)
  = prod_i (1 - P(A_i | earlier bad events did not occur)).
```

For each conditional probability, split the conditioning events into neighbours and non-neighbours.
Non-neighbours disappear by independence; only neighbours must be paid for. Thus the proof turns a
global union-bound sum into a local product over the dependency neighbourhood.

In the asymmetric form, if there are `x_i in [0,1)` such that

```text
P(A_i) <= x_i prod_{j in N(i)} (1 - x_j)
```

for every event, then

```text
P(no A_i occurs) >= prod_i (1 - x_i) > 0.
```

This is why the lemma is so powerful in combinatorics. For hypergraph two-colouring, the union bound
limits the total number of edges. The local lemma instead limits how many other edges each edge meets.
For `k`-SAT, it limits how many clauses share variables with a clause, not the total number of
clauses. The proof method moves from global pessimism to local dependency structure.
