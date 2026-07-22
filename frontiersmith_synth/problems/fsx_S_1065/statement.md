# Demolish the Bridge City, Pier by Pier

## Problem
An old elevated city was built on `n` bridge piers. Every pier still standing is
linked to some other piers by direct structural braces (the brace pattern is
symmetric: if pier `u` braces pier `v`, `v` also braces `u`). A demolition crew
must remove every pier, one at a time, until none remain.

Removing a pier that currently still has `d` active braces to standing piers costs
exactly `d*(d+1)/2` scalar crew-operations: `d` operations to release those braces,
plus `d*(d-1)/2` operations to install a temporary cross-brace between every pair
of those `d` piers that is not already directly braced (so the rest of the
structure stays rigid while work continues). Any cross-brace installed this way is
now a real brace for the rest of the demolition — later removals see it too. This
is exactly the fill-in of symbolic Gaussian elimination applied to the brace
pattern, with the demolition order playing the role of the pivot order.

The city's brace pattern was engineered decades ago as a nested hierarchy of load
groups joined through dedicated coupling piers, then the whole numbering was
scrambled and a handful of long-range emergency braces were added later — so nothing
in the input tells you which piers were originally "load-bearing hubs" versus
ordinary piers. You must find a demolition order minimizing total crew-operations.

## Input (stdin)
```
n m
u_1 v_1
...
u_m v_m
```
`n` piers numbered `1..n`; `m` symmetric braces, each `1 <= u_i, v_i <= n`,
`u_i != v_i`, no brace repeated.

## Output (stdout)
A permutation of `1..n`: the demolition order (whitespace/newlines both fine).

## Feasibility
The output must contain exactly `n` integer tokens, each in `[1, n]`, with no
repeats. Any parse error, wrong count, out-of-range or repeated pier id, or
non-integer token (including `nan`/`inf`) scores `0`.

## Objective
Simulate the demolition in the order given. Maintain the live brace graph,
starting from the input pattern. Process piers in the submitted order; for each
pier still standing with `d` currently-live braces, add `d*(d+1)/2` to the total
op count, then permanently add a brace between every pair of its currently-live
neighbors that isn't already braced (fill-in), then remove the pier. Minimize the
final total op count.

## Scoring
Let `B` be the op count of the naive order `1, 2, ..., n` (piers demolished in
their given numbering, ignoring the brace pattern completely) — the checker
computes this itself. With `F` the op count of your submitted order:

```
Ratio = min(1, 0.1 * B / F)
```

The naive order always scores `0.1`. Halving your op count doubles the ratio.
Because the true minimum fill-in order is NP-hard to compute exactly and the
hidden hierarchy is deliberately obscured, comfortable headroom remains above
what any of the reference strategies below reach.

## Constraints
- `25 <= n <= 60`, `n-1 <= m <= 7n`.
- Deterministic scoring; no timing or GPU involved anywhere.

## Example
Suppose `n=4` with braces `(1,2),(2,3),(3,4)` (a path). Naive order `1,2,3,4`:
removing pier 1 (`d=1`) costs `1`; pier 2 now only touches pier 3 (`d=1`, since
pier 1 is gone) costs `1`; pier 3 touches pier 4 (`d=1`) costs `1`; pier 4 costs
`0`. `B = 3`, no fill-in ever needed on a path. A submission `2,3,1,4` also never
creates fill-in (paths always admit a zero-fill order) and gets `F=3`, so
`Ratio = min(1, 0.1*3/3) = 0.1` — same as naive, since a path has no hidden
hierarchy to exploit. On the real (much denser, hierarchical) instances the gap
between orders that see the hierarchy and orders that don't is large.
