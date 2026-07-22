# Valve Balancing on a Looped Water Trunk

## Problem
A water utility runs a pressure-fed network with one **source** and `n_hub`
**junctions** wired as a trunk: junction `i` (`1..n_hub`) is fed from junction
`i-1` (junction `0` = source, fixed head `H_src`) by **two parallel pipes** — a
trunk pipe and a higher-resistance bypass pipe — so every trunk stage is itself
a small loop. A few extra **cross-tie** pipes connect non-adjacent junctions,
adding longer loops and shortcuts. Exactly `K` junctions are metered
**outlets**, each draining to atmosphere (head `0`) through its own fixed
outlet pipe.

Every pipe obeys the turbulent (quadratic) head-loss law: a pipe of resistance
`R` carrying signed flow `Q` from node `u` to node `v` satisfies
`H_u - H_v = R * Q * |Q|`. At every junction, flow in equals flow out
(Kirchhoff's law); source and atmosphere are the only fixed-head nodes — this
fixes a unique flow for the whole network given the pipe resistances.

You may install a **valve** on any pipe except an outlet's own meter pipe: it
adds extra resistance `x >= 0` in series with that pipe's base resistance.
`x = 0` means untouched (no valve). Using `m` valves costs `lambda_cost * m`,
traded off against how well the settled outlet flows match given targets.
Choose `x` for every valve-capable pipe so that, once the network re-settles,
the `K` outlet flows are close to target while using as few valves as
possible.

## Input (stdin)
```
n_hub K n_edges
H_src lambda_cost
<K outlet junction ids, 1..n_hub>
<K target flows, same order>
n_edges lines: "u v r cap"
```
Node ids: `0`=source, `1..n_hub`=junctions, `n_hub+1`=atmosphere. `r>0` is the
pipe's base resistance; `cap=1` means a valve may be installed, `cap=0` means it
may not (outlet meter pipes only). Edges appear in four fixed blocks, in order:
the `n_hub` **trunk** pipes (pipe `i`, 1-indexed, connects junction `i-1` to
junction `i`), the `n_hub` **bypass** pipes (the same `n_hub` junction pairs,
parallel to the trunk), the cross-tie pipes, then the `K` **outlet** pipes
(always `cap=0`). So `n_edges = 2*n_hub + n_crosstie + K`.

## Output (stdout)
`n_edges` numbers `x_1 .. x_{n_edges}` (any whitespace), the extra resistance
installed on each pipe, in the same order as the input pipe list.

## Feasibility
Every `x_i` must be finite, `0 <= x_i <= 1000`, and `x_i` must equal `0` exactly
(within `1e-6`) whenever pipe `i` has `cap=0`. The output must contain exactly
`n_edges` numbers. Any violation scores `0`.

## Objective (minimize)
Given a feasible `x`, solve the network's flow equations exactly (Newton's
method on the quadratic law above) to get each outlet's settled flow `q_j`.
Let `t_j` be its target. Then:
```
mismatch = sqrt( mean_j (q_j - t_j)^2 )
m        = number of valve-capable pipes with x > 1e-6
F        = mismatch + lambda_cost * m
```
The checker also computes `B`, the same `F` evaluated at `x = 0` everywhere
(no valves at all — the network exactly as built). Score:
```
score = min(1.0, 0.1 * B / max(1e-9, F))
```
Leaving every valve untouched always scores exactly `0.1`; a smaller `F`
(better match, fewer valves) scores higher, up to a cap of `1.0`.

## Constraints
`3 <= n_hub <= 12`, `1 <= K <= n_hub`, `0 <= n_crosstie <= 3`,
`lambda_cost in [0.02, 0.08]`, `H_src = 100`, resistances and targets are
positive floats with modest magnitude. Time limit 5s.

## Example (worked score, illustrative shape only — not one of the test cases)
Take `n_hub=1, K=1`: source `0`, junction `1`, atmosphere `2`. Pipes: trunk
`(0,1,r=2.0,cap=1)`, bypass `(0,1,r=8.0,cap=1)`, outlet `(1,2,r=2.0,cap=0)`.
`H_src=100`, `lambda_cost=0.1`, target `t_1 = 3.0`.

With `x = 0,0,0` (all valves open) the network settles to outlet flow `5.8835`
(head at junction `1` is `76.47`, driven by the parallel 2.0/8.0 pipes), so
`mismatch = 2.8835`, `m=0`, giving `B = 2.8835` (score `0.1` by construction).

Submitting `x = 6.0, 0, 0` (throttle only the trunk pipe) makes the network
settle with junction-1 head `50.0` and outlet flow `5.0`: `mismatch = |5.0-3.0|
= 2.0`, `m = 1`, `F = 2.0 + 0.1*1 = 2.1`. Score `= min(1, 0.1*2.8835/2.1) =
0.1373` — better than doing nothing, but far from optimal (a different `x`
gets closer to `3.0` for the same one valve).
