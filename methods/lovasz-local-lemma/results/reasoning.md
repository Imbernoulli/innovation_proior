I want to prove existence: pick a random object, define bad events `A_1, ..., A_n`, one for each
local way the object can fail, and show `P(no A_i occurs) > 0`. Two tools sit at the extremes. If
the `A_i` are mutually independent, `P(none occurs) = prod_i (1 - P(A_i))`, positive whenever every
`P(A_i) < 1` — but real random constructions never hand me that, because nearby local failures
share the same coin flips. Drop independence and fall back on the union bound,
`P(some A_i) <= sum_i P(A_i)`, and I need the total probability mass under one — a number that
grows with the sheer count of bad events, even when most of them can't possibly interact. Neither
tool is the right shape for what I keep running into: many bad events, each individually unlikely,
that only "feel" a handful of neighbors.

So my events have a local dependency structure and I want to use that structure directly instead of
collapsing it into one of the two extremes. My first instinct is to build a graph on the events by
joining `i ~ j` whenever `A_i` and `A_j` are correlated, then hope non-adjacent events act as
independent. That instinct is wrong, and three fair coins show it: pick `x_1, x_2, x_3` uniformly
and independently from `{0,1}`, and let `A_i` be the event `x_{i+1} + x_{i+2} = 0` (indices mod 3).
Check any two of these and they factor — pairwise independent. But all three together are not:
knowing `A_1` and `A_2` happened pins down `x_3` completely, which decides `A_3`. So "no edges,
because everything is pairwise independent" is not a valid dependency graph here; the empty graph
would let me claim `A_3` is independent of `{A_1, A_2}`, which is false. What I actually need is
stronger than pairwise: `A_i` independent of the whole set of its non-neighbors jointly, independent
of every Boolean combination of them at once, not just of each one taken singly. That's a real
constraint on which graphs I'm allowed to draw, not a bookkeeping footnote.

Given such a graph — max degree `d`, each `A_i` independent of the whole block of non-neighbor
events — I want `P(no A_i)`, and the chain rule gives it exactly, no approximation yet:

```text
P(no A_i occurs) = prod_i P(A_i does not occur | earlier events did not occur).
```

So it's enough to keep every conditional `P(A_i | some earlier events did not happen)` bounded away
from 1. This is already a different object from what the union bound uses — unconditional `P(A_i)` —
and the whole content of what I'm building is that conditioning on far-away events costs nothing,
because of the block independence I just pinned down.

Fix `A_1` and bound `P(A_1 | A_2...A_n)` by induction on `n`. Let `v_2,...,v_q` be `A_1`'s neighbors,
so `q <= d+1`. Split the conditioning set into these neighbors and the rest:

```text
P(A_1 | A_2...A_n) = P(A_1 A_2...A_q | A_{q+1}...A_n) / P(A_2...A_q | A_{q+1}...A_n).
```

The numerator only grows if I drop `A_2...A_q` from it, and `A_1` is independent of the block
`{A_{q+1},...,A_n}` — all non-neighbors — so the numerator is at most `P(A_1 | A_{q+1}...A_n) =
P(A_1)`. For the denominator, expand the complement: `P(A_2...A_q | rest) = 1 - P(A_2 or ... or A_q
| rest) >= 1 - sum_{i=2}^{q} P(A_i | rest)`, a plain union bound on at most `d` terms, each smaller
than the original conditioning problem, so the induction hypothesis applies to each directly.

Now the induction has to close on itself: I'm bounding a conditional probability using the same
bound I've assumed for smaller ones. Suppose `P(A_i) <= p` for every `i` and try the ansatz "every
conditional is `<= beta`" for some `beta` I get to pick. Then the denominator is `>= 1 - d*beta`, and
the whole ratio is `<= p / (1 - d*beta)`. For the induction to actually reproduce `beta`, I need
`p / (1 - d*beta) <= beta`, i.e. `p <= beta*(1 - d*beta)`. That right-hand side is a downward
parabola in `beta`, maximized at `beta = 1/(2d)`, where it equals `1/(4d)`. So `p <= 1/(4d)` is
exactly the largest threshold for which some `beta` makes the induction self-consistent, and that
`beta` is forced to be `1/(2d)`. The constant isn't chosen to look clean — it's the fixed point of
the recursion, and it forces the exact pair that has to appear together:

```text
P(A_i) <= 1/(4d)  =>  P(no A_i occurs) > 0.
```

Applied to a `k`-uniform hypergraph two-coloring: color each vertex red or blue independently with
probability 1/2, let `A_e` be the event that edge `e` comes out monochromatic, so
`P(A_e) = 2 * (1/2)^k = 2^{1-k}`, and build the dependency graph on edges by "shares a vertex."
If the maximum such degree `d` satisfies `2^{1-k} <= 1/(4d)`, i.e. `d <= 2^{k-1}/4`, a proper
2-coloring exists — and that bound is entirely about how many other edges a single edge can meet,
never about the total number of edges. That is the payoff: the classical threshold for this kind of
result is a bound on the total edge count of the whole hypergraph, and this local form replaces it
with a bound on the local meeting-degree, which can stay small forever even as the hypergraph grows
without limit.

But the induction step above is wasteful in a specific, fixable way. The denominator bound,
`1 - sum P(A_i | rest)`, is a union bound applied to the block `{A_2,...,A_q}` — the exact crude
tool I was trying to escape in the first place, just demoted one level, from "all `n` events" down
to "the `d` neighbors of one event." And it treats every neighbor identically, at the same
worst-case threshold `beta`, even though different bad events can carry very different
probabilities. Both problems have the same fix: don't bound "none of the neighbors occur" with a
union bound, bound it with the same chain-rule trick, recursively, one neighbor at a time. Write
`S_1 = {j_1,...,j_r}` for the neighbors still in the conditioning set and expand

```text
P(not A_{j_1} ... not A_{j_r} | rest) = prod_t P(not A_{j_t} | not A_{j_1}...not A_{j_{t-1}}, rest),
```

bounding each factor by the induction hypothesis on a strictly smaller conditioning set — which
means the induction now has to run on the size of the conditioning set, not on `n`, since peeling
off one neighbor at a time is what shrinks the problem. Let each event carry its own slack
`x_i in [0,1)` instead of one shared `beta`, so the hypothesis becomes "`P(A_i | any subset of
non-neighbors fixed) <= x_i`," and the same split as before gives

```text
P(A_i) <= x_i * prod_{j in N(i)} (1 - x_j)  =>  P(no A_i occurs) >= prod_i (1 - x_i) > 0.
```

This dominates the earlier bound because `prod (1 - x_j) >= 1 - sum x_j` — a product is a strictly
better lower bound on "none of finitely many small things happen" than a union-bound complement —
and it lets me tune `x_i` per event instead of forcing one global `beta` on every neighbor
regardless of how likely it actually is.

To pull a clean symmetric statement back out of this, set `x_i = 1/(d+1)` uniformly, the natural
choice when every event has degree `<= d` and probability `<= p`. The condition becomes
`p <= (1/(d+1)) * (1 - 1/(d+1))^d`, and `(1 - 1/(d+1))^d` is an instance of `(1-1/m)^{m-1}` with
`m = d+1`, which I know stays above `1/e` for every finite `m` and decreases down to it as `m` grows
(check `m=2`: `0.5 > 1/e`; the sequence only gets closer to `1/e` from above, never below it). So

```text
e * p * (d+1) <= 1  =>  P(no A_i occurs) > 0,
```

and the `e` is not a rounding artifact — it's the exact price of trading per-event slack for one
uniform slack spread across a degree-`d` block, the same exchange rate that showed up as `4` in the
cruder induction, now tightened to its natural constant by replacing the union bound with the
telescoping product. Revisiting the hypergraph coloring example: the condition is now
`e*(d+1) <= 2^{k-1}`, i.e. `d <= 2^{k-1}/e - 1`, which allows roughly `4/e ≈ 1.47` times as many
edges meeting a given edge as the crude `2^{k-1}/4` bound did, for the same `k`. For `k`-SAT under
this sharper form, a clause with `k` literals is violated by exactly one of `2^k` assignments, so
`P(A_i) = 2^{-k}`; if every clause shares a variable with at most `d <= 2^k/e - 1` other clauses,
then `e * 2^{-k} * (d+1) <= 1` and a satisfying assignment exists — again a statement about local
variable-sharing between clauses, never about the total number of clauses in the formula.
