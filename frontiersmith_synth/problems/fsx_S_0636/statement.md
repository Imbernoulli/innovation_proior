# Spill-Aware Register Coloring on a Budget

## Story

You are given an **interference graph** `G = (V, E)`: `n` values that are live at
overlapping points in a program, with an edge between any two values that must
**not** share the same physical register. Only `k` registers (colors `0..k-1`)
are available. Every value `i` carries a **spill weight** `weight[i] >= 1`: the
runtime cost you pay if `i` cannot be given a register and must be spilled to
memory instead.

You must assign each value either a color in `[0, k)` or leave it **spilled**
(no color). The assignment is **proper** iff no edge joins two values holding the
*same* color (two spilled values never conflict with anything). Your goal is to
choose which values to spill, if any must be, so as to **minimize the total spill
weight** of the spilled values.

The tension: cheap, densely interfering clusters of values look "urgent" (they
run out of usable colors fast) but are cheap to spill if you must; a single
expensive value tangled up with such a cluster can be catastrophic to spill. A
plan that colors greedily by local urgency alone, ignoring cost, can end up
sacrificing the most expensive values instead of the cheapest ones.

## Input (public instance, one JSON object on stdin)

```json
{
  "name": "case7",
  "n": 12,
  "k": 4,
  "weights": [3, 1, 250, 4, ...],
  "edges": [[0, 1], [0, 2], [1, 3], ...]
}
```

- `n` (int): number of values, indexed `0..n-1`.
- `k` (int): number of available colors (registers).
- `weights` (list of `n` ints `>= 1`): spill weight per value.
- `edges` (list of `[u, v]`, `u < v`): undirected interference edges.

## Output (one JSON object on stdout)

```json
{"colors": [c_0, c_1, ..., c_{n-1}]}
```

- Exactly `n` entries, one per value in index order.
- Each `c_i` is an integer, either `-1` (spilled) or in `[0, k)` (a color).
- **Properness**: for every edge `(u, v)`, it must NOT be that
  `c_u == c_v != -1`.

Any of the following makes the instance score `0.0`: wrong length, a
non-integer or out-of-range entry, a properness violation, a crash, a timeout,
or output that is not the JSON object above.

## Objective and scoring (deterministic)

For each instance the evaluator computes two references, both from the graph
and weights alone (never from your output):

- `S_weak`: the spill weight of a fixed **weak baseline** coloring -- process
  values in raw index order `0..n-1`, each taking the smallest color not yet
  used by an already-colored neighbor, spilling if none is free. This ignores
  weight entirely.
- `LB`: a **lower bound** on the best possible spill weight, obtained by
  growing a greedy clique from every value; any clique `C` with `|C| > k`
  forces at least `|C| - k` of its members to spill, and the cheapest such
  forced subset (the `|C|-k` lowest-weight members of `C`) lower-bounds the
  true minimum. `LB` is the best bound found over all probed cliques (this is
  generally a *loose* bound, so even an optimal coloring will usually not
  reach a normalized score of `1.0`).

Let `S_cand` be the spill weight of your (validated) coloring. The instance
score is

```
r = clamp( 0.1 + 0.85 * (S_weak - S_cand) / max(1e-9, S_weak - LB), 0, 1 )
```

Matching the weak baseline scores about `0.1`; a smaller spill scores higher
(up to `0.95` if you match the lower-bound probe exactly, which is rarely
achievable); a larger spill scores lower, down to `0`. Your final score is the
mean of `r` over all instances -- a mix of easy sanity cases, generic
medium-density graphs, and harder cases where a dense, cheap cluster of values
sits next to one or more much more expensive values it interferes with.

## Notes

- Scoring never depends on wall-clock time; treat the time limit as a generous
  compute budget.
- Your program runs in an isolated subprocess and sees only the public
  instance above -- no evaluator internals, no other instances.
- Think about *which* values are cheapest to sacrifice, not just which values
  are locally hardest to color.
