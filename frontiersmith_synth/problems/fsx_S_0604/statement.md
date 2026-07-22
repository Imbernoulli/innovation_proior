# Derivative Mountain Range: Cross-Country Jacobian Accumulation

You are handed a computational graph and must chart the cheapest route through it —
the elimination order that accumulates its full Jacobian with the **fewest scalar
multiplications**.

## The model

A directed acyclic graph has `V` vertices. The first `M` vertices (`0 .. M-1`) are
**independent inputs** (no incoming edges); the last `N` vertices (`V-N .. V-1`) are
**dependent outputs** (no outgoing edges); the rest are **intermediate** vertices.
Each edge `u -> v` carries a local partial derivative `a[u][v]`, a nonzero integer
modulo `P = 1000000007`. The full Jacobian entry for output `o` w.r.t. input `i` is
the sum over all directed paths `i -> o` of the product of the edge derivatives on
the path (all arithmetic in `GF(P)`).

## Vertex elimination

The Jacobian is accumulated by **eliminating** intermediate vertices one at a time.
Eliminating a vertex `v` with predecessor set `Pred(v)` and successor set `Succ(v)`:

- for every `p in Pred(v)` and every `s in Succ(v)`, update
  `a[p][s] += a[p][v] * a[v][s]` (creating the edge `p -> s` if absent);
- then delete `v` and its incident edges.

This costs exactly `|Pred(v)| * |Succ(v)|` **multiplications**. Eliminating every
intermediate vertex (in any order) leaves only input→output edges — those edges are
the Jacobian, and it is the same Jacobian for every order. **Only the total
multiplication count depends on the order.**

## Your task

Output a permutation of the intermediate vertices `{M, .., V-N-1}` — the order in
which to eliminate them. Fewer total multiplications is better.

Forward mode (eliminate in topological order) and reverse mode (reverse topological)
are the two textbook answers; both carry a full input- or output-sized frontier
across the whole graph. The graph is deliberately shaped with **narrow cuts**
(bottleneck layers every path must funnel through). Pre-accumulating *around* each
cut — collapsing both sides down to the bottleneck before crossing it — can beat
either pure mode several-fold. The cut positions and widths are not stated; you must
discover them from the edge list.

## Input (stdin)

```
V E M N
u_1 v_1 w_1
...
u_E v_E w_E      (edge u->v with derivative w, 1 <= w < P)
```

## Output (stdout)

Whitespace-separated integers: a permutation of `{M, .., V-N-1}` (each intermediate
vertex exactly once).

## Feasibility

Any of these makes your score `0`: a token that is not an integer; an id outside the
intermediate range; a repeat; or a set that is not exactly the intermediate vertices.

## Scoring

The checker replays your order in `GF(P)`, verifies the resulting input→output
matrix equals the true Jacobian, and counts your multiplications `F`. Let `B` be the
checker's forward-mode multiplication count. Your ratio is

```
Ratio = min(1.0, 0.1 * B / F)
```

so forward mode scores about `0.1`, and driving the op count `10x` below forward
caps at `1.0`. Reported per case over 10 cases; deterministic.

## Example

For a graph inputs→A→(single waist w)→B→outputs where `A`, `B` are wide, forward
mode drags the full input dimension through both wide layers, while collapsing `A`
into `w` and `B` out of `w` first makes the final cross-cut multiplication tiny.
Same Jacobian, far fewer multiplications.

## Constraints

`V <= 500`, edges fit well under 5 MB, time limit 5 s, memory 512 MB. All values and
scoring are exact integers; the score is fully deterministic.
