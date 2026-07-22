# Saturating Sensor Placement with Critical Double-Cover Bonuses

## Problem
A facility network has `N` nodes connected by weighted edges (all weights positive
integers); `dist(u,v)` is the shortest-path distance. Each node `v` has an **importance**
`w(v) >= 0`, a **saturation scale** `tau(v) >= 1`, a flag `crit(v) in {0,1}` marking whether
it is *critical*, and a **double-cover bonus** `beta(v) >= 0` (only meaningful when
`crit(v)=1`).

You must choose a **budget** of exactly `K` distinct nodes to host sensors (any node may
host one). A sensor at site `s` **reaches** node `v` only if `dist(s,v) <= R` (a fixed
reach threshold `R` given in the input); reaching contributes `reach(s,v) = R - dist(s,v)`
to `v`'s accumulated coverage **mass**. Let `cov(v)` be the number of *distinct* placed
sensors that reach `v`, and `mass(v)` the sum of their `reach` values.

Each node turns its mass into value through a **saturating** (diminishing-returns) curve,
and critical nodes additionally earn a flat bonus the moment a **second** distinct sensor
reaches them:
```
base(v)  = w(v) * (1 - exp(-mass(v) / tau(v)))
bonus(v) = beta(v)   if crit(v)=1 and cov(v) >= 2,  else 0
```
Your goal is to **maximize** `F = sum_v ( base(v) + bonus(v) )`.

Because `bonus(v)` only appears once `cov(v)` reaches 2, a critical node's *second*
reaching sensor can be worth much more than its first -- the objective is **not**
submodular, and blindly chasing whichever site currently looks best, one at a time, can
permanently miss these bonuses.

## Input (stdin)
```
N M K R
```
Then `N` lines, one per node `v = 0..N-1`:
```
w(v) crit(v) tau(v) beta(v)
```
Then `M` lines, one per edge (undirected, 0-indexed endpoints, positive integer weight):
```
u v weight
```

## Output (stdout)
Exactly `K` integers (space/newline separated): the ids of the chosen sensor sites,
each in `[0, N)` and **pairwise distinct**.

## Feasibility
- Exactly `K` tokens, each parses as a finite integer.
- Every id lies in `[0, N)`.
- No id repeats (sensors occupy distinct sites).

Any violation scores `Ratio: 0.0`.

## Scoring
Let `F` be your objective value (above). The checker also computes `B`, the value of the
**trivial** placement that puts all `K` sensors on node ids `0, 1, ..., K-1` (ignoring the
graph and every weight/bonus). Then
```
sc    = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000
```
so matching the trivial placement scores about `0.1`; a placement worth ten times more
than the trivial baseline caps the score at `1.0`.

## Constraints
`15 <= N <= 50`, `1 <= K <= N`, `R = 3` (fixed), edge weights and `tau(v) >= 1` are
positive integers; `0 <= w(v) <= 5`, `0 <= beta(v) <= 20`. Ten test cases grow from small to large;
larger cases plant several critical nodes whose two best-reaching sites are never both the
locally best single choice, so a purely myopic strategy tends to reach many nodes once each
and never trigger any double-cover bonus.

## Example
Take a tiny instance: nodes `0` (critical, `w=4, tau=3, beta=8`), `1` and `2` (helpers,
`w=0`), `3` (a decoy, `w=5, tau=1`), `R=2`, edges `0-1` (weight 1), `0-2` (weight 1), `3`
isolated (all edges to it have weight `> R`), `K=2`.
- Placing sensors at `{1, 2}`: both reach node 0 (`dist=1<=2`), so `cov(0)=2`,
  `mass(0) = (2-1)+(2-1) = 2`, `base(0) = 4*(1-e^{-2/3}) ≈ 1.95`, plus the bonus
  `beta(0)=8` since `cov(0)>=2`. Nodes `1,2,3` contribute `0`. Total `F ≈ 9.95`.
- Placing sensors at `{0, 3}`: node 0 gets `cov(0)=1` (only itself, `dist=0`),
  `mass(0)=2`, `base(0) ≈ 1.95`, **no bonus** (`cov(0)=1 < 2`). Node 3 gets
  `mass(3)=2`, `base(3) = 5*(1-e^{-2}) ≈ 4.32`. Total `F ≈ 6.27` -- worse, even though
  each single choice (`0` alone, `3` alone) looked locally attractive.
This is exactly the trap: covering node 0 **twice** (via its two helpers) beats spending
one sensor on node 0 and one elsewhere, but neither helper alone looks as good as the
decoy `3`.
