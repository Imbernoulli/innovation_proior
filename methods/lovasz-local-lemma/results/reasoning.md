The starting point is the standard probabilistic-method setup. Choose a random object and define bad
events `A_1, ..., A_n`, one for each local way the object can fail. The existence goal is
`P(no A_i occurs) > 0`. If this probability is positive, at least one outcome avoids every bad event,
so the desired object exists.

There are two simple ways to prove this, and the Lovasz Local Lemma is interesting because it lives
between them. If all bad events are mutually independent and each has probability less than one, then
the probability of avoiding them is the product `prod_i (1 - P(A_i))`, which is positive. This is
strong but rarely available, because local constraints usually share random choices. If independence
is unavailable, the union bound says `P(some A_i occurs) <= sum_i P(A_i)`, so avoiding all bad events
has positive probability when the total sum is below one. This is robust but globally wasteful.

The union bound's weakness is structural. It asks for a global budget over all bad events, even when
most of those events have no statistical relationship to each other. In hypergraph two-colouring, for
example, the bad event that an edge is monochromatic depends only on the colours of vertices in that
edge. A disjoint edge uses disjoint coin flips. Charging every edge in a single global sum ignores
this locality, and it makes the conclusion depend on the total number of edges rather than on how
crowded the neighbourhood of each edge is.

The local lemma's insight is to model dependence itself as the resource. Put a graph on the bad
events, joining `A_i` to the events it may depend on. The graph must mean set-level independence:
`A_i` is independent of the whole collection of non-neighbour events, not merely independent of each
one pairwise. This distinction matters because pairwise independence does not let a conditional
probability ignore a whole block of events. The proof needs exactly that block independence.

The key proof object is not the unconditional probability `P(A_i)`. The exact expansion is the chain
rule:

```text
P(no A_i occurs)
  = prod_i P(A_i does not occur | earlier bad events did not occur)
  = prod_i (1 - P(A_i | earlier bad events did not occur)).
```

So it is enough to show that every conditional probability
`P(A_i | some other bad events did not occur)` stays below a threshold less than one. This is where
the proof departs from the union bound. The union bound sums unconditional probabilities globally;
the local lemma controls conditional probabilities using only the dependency neighbourhood.

For a fixed event `A_i`, split the conditioning set into two parts: neighbours of `A_i`, and
non-neighbours of `A_i`. The non-neighbour part is far away in the dependency graph, so by the
definition of the graph it is independent of `A_i`; it does not inflate the numerator. The neighbour
part remains, but it is small. In the sharp general form, the denominator is expanded by the chain
rule into a product of terms, each bounded below by an induction hypothesis. This is why the final
condition has a product over neighbours:

```text
P(A_i) <= x_i prod_{j in N(i)} (1 - x_j).
```

Here `x_i` is the allowed conditional risk for event `A_i`. If this inequality holds for every `i`,
the induction proves
`P(A_i | any chosen set of other bad events did not occur) <= x_i`. Substituting those bounds back
into the global chain-rule product gives
`P(no A_i occurs) >= prod_i (1 - x_i) > 0`.

The symmetric form is the memorable special case. If every event has probability at most `p` and has
at most `d` neighbours, choose the same slack for every event: `x_i = 1/(d+1)`. Then the local product
is at least `(1/(d+1))(d/(d+1))^d`, and this is at least `1/(e(d+1))`. Therefore the asymmetric
hypothesis is satisfied whenever `e p (d+1) <= 1`.

This is the conceptual shift: the cost of an event is no longer multiplied by the total number of
bad events. It is compared with the size of the event's dependency neighbourhood. The lemma does not
pretend that the bad events are globally independent. It says global independence is unnecessary
when the dependence graph is sparse and every local failure is sufficiently unlikely.

For hypergraph two-colouring, this changes the kind of theorem available. A random red-blue colouring
makes a `k`-edge monochromatic with probability `2^{-(k-1)}`. The union bound can only say something
when the total number of edges is below about `2^{k-1}`. The local lemma instead says that if each
edge intersects only about `2^{k-1}/e` other edges, then a proper two-colouring exists, regardless of
the total number of edges. The same pattern appears in `k`-SAT: the event that a clause is violated
has probability `2^{-k}`, and satisfiability is guaranteed when each clause shares variables with
few enough other clauses. Total clause count is not the controlling parameter; local dependency is.

The original Erdos-Lovasz argument already captures the central move with a cruder constant: if each
event depends on at most `d` others and `P(A_i) <= 1/(4d)`, the chance of avoiding all bad events is
positive. The modern asymmetric statement keeps the same proof anatomy but replaces the local union
bound in the denominator with a product of neighbour slacks. That refinement exposes the real form
of the insight: sparse local dependence can be budgeted multiplicatively, event by event.
