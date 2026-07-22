# Twelve Cantons, One Bottle Market

## Problem
A national bottle-deposit scheme covers `n` cantons (n = 12). Each canton `i` posts its
own deposit `d_i` (an integer number of cents, `0 <= d_i <= D_MAX`) that it pays out for
every bottle redeemed locally. Residents' willingness to return bottles rises with the
posted deposit along a canton-specific, piecewise-linear, concave **elasticity curve**:
below a *dead-zone threshold* `d0_i` nobody bothers; then rate ramps to breakpoint
`(d1_i, r1_i)` and on to `(d2_i, r2_i)` (rates are parts-per-million of the population,
`0 < r1_i < r2_i <= 1000000`, `0 <= d0_i < d1_i < d2_i <= D_MAX`):
```
rate(d) = 0                                                     if d <= d0_i
        = r1_i * (d - d0_i) / (d1_i - d0_i)                     if d0_i < d <= d1_i
        = r1_i + (r2_i - r1_i) * (d - d1_i) / (d2_i - d1_i)     if d1_i < d <= d2_i
        = r2_i                                                  if d > d2_i   (saturated)
```
(integer division, floored). Genuine local returns are `ret_i = floor(pop_i * rate(d_i) / 1e6)`.
The dead-zone `d0_i` differs by canton, so no single deposit level is elasticity-optimal
for every canton at once.

Cantons are linked by a **hauler corridor graph**: `E` undirected edges `(u, v, cost)`,
`cost` in cents. Haulers act deterministically: for every canton `i`, they find the
reachable canton `j` (via the graph's **shortest-path** transport cost `dist(i,j)`,
possibly through several intermediate cantons) that maximizes `d_j - dist(i,j)`. If this
best value exceeds `d_i`, every bottle genuinely returned in `i` is instead redeemed by
haulers at `j`'s deposit: the *effective payout* for canton `i`'s returns becomes `d_j`
instead of `d_i`. Otherwise the effective payout stays `d_i`. (If `i` is unreachable from
any better-paying canton, nothing is diverted.)

## Input (stdin)
```
n V D_MAX F_permille
name_1 pop_1 d0_1 d1_1 r1_1 d2_1 r2_1
...
name_n pop_n d0_n d1_n r1_n d2_n r2_n
E
u_1 v_1 cost_1
...
u_E v_E cost_E
```
`V` is the recycling value (cents) credited per genuine bottle returned, regardless of
where it is redeemed. `F_permille` is the float-cost rate (thousandths) charged against
every cent actually paid out. Cantons are 0-indexed in input order for the edge list.

## Output (stdout)
Exactly `n` whitespace-separated integers `d_0 ... d_{n-1}`, canton `i`'s posted deposit.

## Feasibility
Output must contain exactly `n` tokens, each parsing as a finite integer with
`0 <= d_i <= D_MAX`. Any violation scores `Ratio: 0.0`.

## Objective
For each canton `i`, let `eff_i` be the effective payout described above. Maximize
```
F = sum_i ( ret_i * V - ret_i * eff_i - floor(ret_i * eff_i * F_permille / 1000) )
```
i.e. total recycling value, minus what haulers actually cause the scheme to pay out
(local or diverted), minus float cost on that payout. Posting deposits that violate the
shortest-path no-arbitrage bound (`d_i - d_j > dist(i,j)` for some reachable `j`) does not
make the output infeasible -- it simply bleeds budget to haulers.

## Scoring
Let `B` be the objective `F` obtained from the checker's own fixed reference vector
(every canton posts the same uniform deposit `U = 90`, which is always arbitrage-free
since all deposits are equal). With `F' = max(0, F)`:
```
sc = min(1000.0, 100.0 * F' / max(1e-9, B))
Ratio = sc / 1000.0
```
Matching the uniform baseline scores `0.1`; ten times better caps at `1.0`.

## Constraints
`n = 12`, `1 <= D_MAX <= 400`, populations up to `120000`, `1 <= E <= 35`, edge costs
`>= 1`. Time limit 5s, memory 512MB.

## Example (illustrative shape only, not a real test)
Two cantons, no edge between them (`dist = infinity`, so nothing is ever diverted). Each
posts its own elasticity-optimal deposit independently; `F` is just the sum of the two
standalone canton values, and `B` is the value of both posting `U`. Real tests connect
all 12 cantons through a shared corridor graph, so choosing each canton's deposit in
isolation is generally unsafe.
