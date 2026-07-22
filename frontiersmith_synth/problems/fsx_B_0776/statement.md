# Reservoir Trunk Main: Sizing Pipes Against Quadratic Head Loss

## Problem
A hillside reservoir (node `1`) feeds a **tree-shaped** pipe network down to `n-1`
village nodes (`2..n`), each with a fixed water demand. Every pipe (tree edge) must be
built at one of a small menu of discrete diameters under a total cost budget.

Because the network is a tree, the **flow** through the pipe feeding node `v` is fixed
by demand conservation: it equals the total demand of everything downstream of `v`
(node `v`'s subtree) — sizing decisions cannot change *how much* flow a pipe carries,
only how much **head (pressure) it loses** carrying it. Head loss follows a
Weymouth-style law: for a pipe with flow `Q`, length `L`, roughness `K`, and diameter
`D`, the loss is `K * L * Q^2 / D^5` — quadratic in flow, and steeply (inverse fifth
power) reduced by diameter. A node's available head is the reservoir's head minus the
*sum* of head losses along the whole path down to it, so an under-sized pipe near the
root hurts **every** node behind it, not just its immediate child.

## Input (stdin)
```
n
ndiam D_1 D_2 ... D_ndiam
C
parent_2 demand_2 length_2 unit_cost_2 K_2
parent_3 demand_3 length_3 unit_cost_3 K_3
...
parent_n demand_n length_n unit_cost_n K_n
```
`n` nodes, node `1` is the reservoir (source, no demand). `ndiam` ascending diameters
are available to every pipe. `C` is the total cost budget. For each node `i` (`2..n`,
in order), `parent_i < i` is its supply node, `demand_i >= 1` its water demand,
`length_i` and `unit_cost_i` are pipe-length and per-unit-length cost, and `K_i` the
roughness coefficient of the pipe feeding it. Building pipe `i` at diameter `D`
costs `unit_cost_i * length_i * D^2`.

## Output (stdout)
`n-1` integers `idx_2 idx_3 ... idx_n` (any whitespace layout), each an index into the
diameter menu (`0`-based) — the diameter chosen for the pipe feeding node `i`.

## Feasibility
- exactly `n-1` integer tokens, each finite and in `[0, ndiam-1]`;
- total cost `sum(unit_cost_i*length_i*D_{idx_i}^2) <= C`.
Any violation scores `Ratio: 0.0`.

## Objective (minimize)
For node `v`, let `Q_v` = subtree demand (flow through its pipe), and let `L(v)` be
the sum of head losses along the path from the reservoir to `v` under your diameter
choices. Let `bestL(v)` / `worstL(v)` be that same path-loss sum if *every* pipe on the
path used the *largest* / *smallest* menu diameter respectively (these only depend on
the input, not on your answer — they calibrate how much control diameter choice can
possibly have over that specific node). Node `v`'s satisfied fraction is
```
f(v) = clamp( (worstL(v) - L(v)) / (worstL(v) - bestL(v)), 0, 1 )
```
(if `worstL(v) == bestL(v)`, diameter can't change that node's outcome, so `f(v)=1`).
Unmet demand at `v` is `(1 - f(v)) * demand_v`. Minimize `F = sum_v (1 - f(v)) * demand_v`.

## Scoring
The checker builds its own baseline `B`: the largest diameter usable **uniformly on
every pipe** while staying within budget `C`, scored the same way as above. Then
```
sc = min(1000.0, 100.0 * B / max(1e-9, F))
Ratio = sc / 1000.0
```
Matching the uniform baseline scores `0.1`; a construction with `10x` less unmet
demand caps at `1.0`.

## Constraints
- `5 <= n <= 80`, `ndiam = 8`, diameters `1..8`.
- `1 <= demand_i <= 6`, `5 <= length_i <= 20`, `1 <= unit_cost_i <= 3`, `0.5 <= K_i <= 1.5`.
- `C` is chosen per test so the budget is genuinely tight (never enough to give every
  pipe the largest diameter).
- Time limit 5s, memory 512m.

## Example
One pipe: `n=2`, diameters `1 2 3 4`, `C=40`, node `2`: parent `1`, demand `5`,
length `10`, unit_cost `1`, `K=1.0`. Costs are `10,40,90,160`; only `D=1` or `D=2`
fit `C=40`. `worstL` (D=1) `=250`; `bestL` (D=4, unaffordable, scale-only) `≈0.244`.
`D=2` gives `L≈7.81`, `f≈0.970`, unmet `≈0.152`. The uniform baseline is stuck at
`D=1` (`f=0`, `B=5`), so `sc=min(1000,100*5/0.152)=1000`: this toy trivially
saturates with one pipe — real instances have many pipes sharing one budget, so no
single choice saturates the score (see Scoring above).
