# Relay-Clock Sync: Rationed Cable Upgrades Under Node Caps

A distributed relay-clock network has `n` stations `0..n-1`, connected by `m`
already-installed cables. Every cable currently carries weight 1 (its
consensus-coupling strength). The stations run a standard diffusive
consensus protocol, whose convergence rate to a common clock is governed by
`lambda_2`, the second-smallest eigenvalue of the network's weighted
Laplacian `L = D - A` (`D` weighted-degree diagonal, `A` weighted
adjacency). Larger `lambda_2` means faster, more robust sync.

You have an integer upgrade budget `W`: you may raise any subset of cables'
weights by non-negative integer amounts, spending at most `W` total extra
weight units. But every station `i` has a **weighted-degree cap** `cap[i]`:
the sum of the weights of all cables touching station `i` (after upgrade)
must never exceed `cap[i]`. Caps vary station to station — some stations
already sit right at their cap (no spare capacity at all), others have
generous spare capacity. You must read `cap[]` from the input; it is not
implied by the statement.

**The catch.** It is tempting to find "the" cable that most limits sync
speed and pour the whole budget into it. But `lambda_2` is a property of
the *whole* network, and the cable that looks most critical is frequently
one whose endpoints are already at (or nearly at) their cap — upgrading it
is capped almost immediately, no matter how much budget you have. Because
`lambda_2(w) = min_{x perp 1, ||x||=1} sum_e w_e (x_u-x_v)^2` is a concave
function of the weight vector, there is no single formula for the optimal
spend: raising one cable's weight shifts which cable matters most next
(diminishing returns, cap saturation elsewhere). A sound allocation has to
be discovered per instance from the concrete graph and cap array.

## Input (stdin)
```
n m W
u_1 v_1
...
u_m v_m
cap_0 cap_1 ... cap_{n-1}
```
`0 <= u_i, v_i < n`, no self-loops, no repeated `{u,v}` pair; the base graph
(all weights 1) is connected. `cap[i] >= (base weighted degree of i)` for
every station, so the all-weight-1 configuration is always feasible.

## Output (stdout)
`m` lines; line `e` (1-indexed, matching the input cable order) is a single
integer `w_e >= 1`, the final weight of that cable.

## Feasibility
- Exactly `m` integer tokens, each finite and `>= 1`.
- `sum_e (w_e - 1) <= W` (total extra weight spent).
- For every station `i`: `sum_{e touching i} w_e <= cap[i]`.
Any violation scores `Ratio: 0.0`.

## Objective (maximize)
`F` = `lambda_2` of the weighted Laplacian built from your `w_e` values.

## Scoring
The checker builds its own reference weighting `B` — a naive round-robin
water-filling of the full budget `W` over the cables in a fixed
(input-seeded) order, respecting the same caps, with no spectral reasoning
— and computes `B`'s `lambda_2`. It reports
```
Ratio = min(1.0, 0.1 * F / B)
```
(so matching the naive baseline scores `0.1`; ten times better caps at
`1.0`). The reference deliberately leaves real headroom above it.

## Constraints
`n <= 26`, `m <= ~200`, `W` given per instance (small relative to the
network, a genuine rationing constraint), time limit 5s, memory 512 MB.

## Example
Path `0-1-2` (`n=3, m=2, W=2`), cables `(0,1)` and `(1,2)`,
`cap = [2, 4, 2]` (each of the 3 stations has exactly 1 unit of spare
weighted-degree beyond its base load). Base `lambda_2 = 1.0`. Output
`2` / `2` (both cables raised to weight 2) spends the full budget
`(2-1)+(2-1)=2=W` without breaking any cap, and raises `lambda_2` to
`2.0`, doubling the sync rate.
