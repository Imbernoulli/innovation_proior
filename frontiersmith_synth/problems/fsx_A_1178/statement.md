# Root-Anchored Loss Tomography

## Problem

A network operator owns a routing tree rooted at a monitoring station
(node `0`). Every link (edge) `e = (parent(v), v)` has a hidden, non-negative
"log-loss" cost `c_v` (larger means lossier). For a handful of nodes the
operator has run an end-to-end probe from the root and recorded the *exact*
cumulative cost `D(v) = sum of c_u over every edge u on the root-to-v path`.
No other information about individual links is available.

Your job: output a non-negative cost for **every** edge of the tree that is
consistent with all reported probes, choosing values that are as close as
possible to the true (hidden) per-edge costs.

The catch: with only a subset of nodes probed, a stretch of unprobed edges
lying between two probed checkpoints has its **total** cost pinned exactly by
the two cumulative readings, but the way that total splits among the
individual edges of the stretch is *not* determined by the data at all — many
different splits are equally consistent with every probe. Some edges are
touched by no probe whatsoever, so not even their sum is constrained.
Reporting *some* feasible split is easy; reporting one that is actually close
to the truth requires reasoning about what the topology can tell you even
where the probes cannot.

## Input (stdin)

```
testId N M
p_1 p_2 ... p_{N-1}
v_1 D_1
...
v_M D_M
```
`N` is the number of nodes (`0..N-1`, root = `0`); `p_i` is the parent of
node `i` (`1 <= i <= N-1`, always `p_i < i`). Then `M` probe lines follow,
each giving a probed node `v` and its exact cumulative root-to-`v` cost
`D_v` (6 decimal places). Every probed node other than ones on a shared
root-to-checkpoint stretch is otherwise unconstrained by the others.
`testId` may be ignored by your solution; it only lets the checker
regenerate the instance.

## Output (stdout)

`N-1` non-negative real numbers, one per line, in order: the cost you assign
to the edge `(p_i, i)` for `i = 1..N-1`.

## Feasibility

For every probed pair `(v, D_v)`, the sum of your reported edge costs along
the root-to-`v` path must equal `D_v` within `1e-3`. All reported costs must
be finite and non-negative. Any violation scores `0`.

## Objective & Scoring

Let `true_v` be the hidden true cost of edge `i` and `pred_v` your reported
value. Per-edge closeness is `acc_v = max(0, 1 - |pred_v - true_v| / scale)`
where `scale` is the instance's mean true edge cost. Your raw score is the
mean of `acc_v` over all `N-1` edges. The checker also builds its own
naive-but-feasible reference (dump each ambiguous stretch's whole sum onto
its root-side edge, guess a flat constant for untouched edges) to get a
baseline `B`, and reports `Ratio = min(1, F / (10*B))`.

## Constraints
`8 <= N <= 60`, `1 <= M <= N-1`, edge costs and cumulative values fit in
`[0, 2000]`. Time limit 3s, memory 512MB.

## Example (illustrative shape only — not an actual generated case)

Tree: `1`'s parent is `0`; `2` and `3`'s parent is `1`; `4` and `5`'s parent
is `3`. True costs (hidden): edge1=5, edge2=3, edge3=4, edge4=2, edge5=1.6.
Probes given: `(1, 5.0)`, `(2, 8.0)`, `(4, 11.0)`.

Node `1`'s probe pins edge1=5 exactly (root anchor). Node `2`'s probe pins
edge2=3 exactly (its parent `1` is also probed). Node `4`'s probe only tells
you edge3+edge4 = 11-5 = 6 — a two-edge stretch with one equation. Edge5
is touched by no probe at all.

An *equal-split* answer would report edge3=edge4=3, and might default edge5
to some flat guess — plausible but structurally uninformed. Reading the
given topology, node `3`'s subtree carries 2 leaves (`4` and `5`) while
node `4`'s subtree carries 1 leaf; splitting the pinned sum of 6
proportionally to those leaf counts (2:1) gives edge3=4.0, edge4=2.0 —
exactly the hidden truth here. The same leaf-count law, calibrated from the
edges that *are* fully pinned (edge1, edge2), extrapolates to a much better
guess for the untouched edge5 than an arbitrary constant would.
