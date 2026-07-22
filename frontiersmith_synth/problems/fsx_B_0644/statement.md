# Buy Redundancy So The Link Never Drops

## Problem
A network of `n` nodes and `m` directed-cost, undirected-topology links connects
`k` designated **terminal** nodes. Each edge `e` independently fails with a
known base probability `p0(e)`; when an edge fails it is simply gone for that
draw. You control a maintenance **budget** `B`. For every edge you may buy 0,
1, 2, ... up to a per-edge cap `maxu(e)` discrete **reliability upgrades**;
each upgrade **halves the edge's remaining failure probability**
(after `u` upgrades, edge `e` fails with probability `p0(e) * 0.5^u`). The
`u`-th upgrade on edge `e` costs a fixed `cost(e)` budget units (so buying all
the way to level `u` costs `u * cost(e)`); you may spend at most `B` in total
across all edges, and edges fail/survive independently once your upgrades are
fixed.

You must decide, for every edge, how many upgrades to buy so as to **maximize
the probability that all `k` terminals end up in the same connected
component** once every edge has independently resolved alive or dead.

This objective is *not* a sum over edges and it is *not* about any single
path: a terminal set stays connected only when every "cut" of the network
that separates some terminals from others has at least one surviving edge.
Money spent on an edge that already has plenty of parallel backup (so its cut
is unlikely to fail regardless) buys far less true probability than the same
money spent on an edge that is the *sole* crossing of its cut.

## Input (stdin)
```
n m k B
t_1 t_2 ... t_k
u_1 v_1 p0_1 cost_1 maxu_1
...
u_m v_m p0_m cost_m maxu_m
```
- `n,m` = node/edge counts; `k` = number of terminals; `B` = total budget.
- `t_1..t_k` = the terminal node ids (1-indexed, distinct).
- Each of the `m` edge lines: endpoints `u_i,v_i` (1-indexed), base failure
  probability `p0_i` given as an integer **per mille** (`p0_i/1000`), the cost
  `cost_i` of ONE upgrade on this edge, and the per-edge upgrade cap
  `maxu_i`.
- `2 <= k <= 5`, `2 <= n <= 9`, `2 <= m <= 16`, `1 <= B <= 300`,
  `1 <= p0_i <= 999`, `1 <= cost_i <= 20`, `1 <= maxu_i <= 3`.

## Output (stdout)
Exactly `m` integers `u_1 ... u_m` (whitespace/newline separated, in the same
order the edges were given): `u_i` is the number of upgrades bought for edge
`i`.

## Feasibility
- Exactly `m` integer tokens, each parseable.
- `0 <= u_i <= maxu_i` for every edge.
- `sum(cost_i * u_i) <= B`.

Any violation scores `Ratio: 0.0`.

## Objective
Let `p_i(u_i) = p0_i/1000 * 0.5^{u_i}` be edge `i`'s failure probability under
your allocation. Each edge resolves independently alive (prob `1-p_i`) or
dead (prob `p_i`). Maximize
`F = Pr[ all k terminals lie in the same connected component ]`,
computed exactly over the `2^m` independent edge outcomes.

## Scoring
The checker computes `F` for your allocation and `B_ref` for the always-legal
"spend nothing" allocation (`u_i = 0` for all `i`), then reports
`Ratio = min(1000, 100*F/B_ref) / 1000`. Doing nothing scores near `0.1`;
a materially better allocation scores higher, capped at `1.0`.

## Example
`n=3 m=2 k=2`, terminals `1 3`, budget `B=10`:
```
1 2 500 5 2
2 3 500 5 2
```
Two edges in series, each starts with `p0=0.5`, cost 5 per upgrade, cap 2.
Spending nothing: `F = 0.5*0.5 = 0.25 = B_ref`. Buying one upgrade on EACH
edge (cost 5+5=10, exactly the budget) gives each edge failure `0.25`, so
`F = 0.75*0.75 = 0.5625`, more than double `B_ref` — better than dumping both
upgrades into a single edge (`0.5*0.875 = 0.4375`), because with only one
route, both links are equally irreplaceable.

## Constraints
Time limit: 5 seconds per test. Memory: 512 MB.
