# Fast-Pass Pricing on the Far Side of the Crowd

## Problem
A theme park sells a **fast pass** for each of `M` rides. Every ride has a **regular lane**
and a **fast lane**. You set the fast-pass **price** `p_j >= 0` for each ride `j`. Then every
one of the `N` visitors decides, on their own, which passes to buy. Waits depend on how many
people end up in each lane, so your prices and the visitors' choices settle into a
**self-consistent equilibrium**. You are paid from pass revenue and rewarded for the surplus
you leave guests.

### Waits (crowding feedback)
Let `n_j` be the number of visitors whose itinerary includes ride `j`. If `F_j` of them hold a
fast pass for ride `j`, the remaining `n_j - F_j` ride the regular lane, and the waits are
```
w_reg_j  = (n_j - F_j) / s_reg_j
w_fast_j =        F_j  / s_fast_j
Savings_j(F_j) = w_reg_j - w_fast_j       (time a pass saves, at F_j buyers)
```
`s_fast_j > s_reg_j`. **As more people buy the pass, the fast lane fills and the savings shrink.**

### Who buys (self-selection fixed point)
Each visitor `i` has a time-value `v_i` and will juggle **at most `K_i` fast passes**. Treating
the waits as given, they value ride `j`'s pass at `net_ij = v_i * Savings_j(F_j) - p_j` and buy
the up-to-`K_i` rides in their itinerary with the **largest positive** `net_ij`. This is applied
as best-response in visitor order `0,1,...,N-1` for `T` sweeps (a deterministic fixed point;
it is a congestion game and converges). Final buyer counts `F_j` and savings follow.

## Input (stdin)
```
M N LAM T
s_reg_1 s_fast_1 ref_1
...
s_reg_M s_fast_M ref_M
v_1 K_1 k_1 r_1 ... r_{k_1}
...                                  (N visitor lines; r's are 1-based ride ids)
```
`LAM` encodes `lambda = LAM/1000`. `ref_j` are posted reference prices (see Scoring). `T` sweeps.

## Output (stdout)
`M` lines: the price `p_j` for each ride (real, `0 <= p_j <= 1e12`), in ride order.

## Feasibility
Exactly `M` finite, non-negative prices. Any missing/extra/negative/non-finite token scores
`Ratio: 0.0`.

## Objective (maximize)
At the induced equilibrium,
```
Revenue = sum_j p_j * F_j
Surplus = sum_j ( (sum of v_i over buyers of j) * Savings_j(F_j) - p_j * F_j )
score   = Revenue + lambda * Surplus
```

## Scoring
The checker computes the equilibrium for your prices and for the posted reference prices
`ref_j`, obtaining `score` and a baseline `B` (the reference `score`). With maximization
normalization:
```
sc    = min(1000, 100 * score / max(1e-9, B))
Ratio = max(0, min(1, sc / 1000))
```
Reproducing the reference prices scores `Ratio = 0.1`; a `10x`-better plan caps at `1.0`.

## Constraints
`3 <= M <= 30`, `N <= 7000`, `1 <= K_i`, `2 <= s_reg_j < s_fast_j`, `0 <= lambda <= 1`, `T <= 45`.
Time limit 5s, memory 512m.

## Why it is hard
The naive move is to read a **static demand curve**: at price `p`, "who is willing?" using the
uncrowded savings, then take the revenue-maximizing `p`. That curve makes the large casual
crowd look lucrative and points you to a **low** price — but a low price floods the fast lane,
`Savings_j` collapses, and most of those "willing" buyers evaporate in the equilibrium. The
revenue-optimal price sits on the **far side of the crowding fixed point**: price the lane to
stay **scarce** in the induced equilibrium, well above where any static analysis points.

## Example
Two visitors want ride `1` with `s_reg=2, s_fast=4`, values `v=10` and `v=1`, `K=1`, `lambda=0`.
At price `p=5`: if only the high-value guest buys, `F=1`, `Savings=(1)/2-(1)/4=0.25`, their
`net = 10*0.25-5 = -2.5 < 0`, so nobody buys and revenue is `0`. At price `p=2`: the high-value
guest buys (`F=1`, `net = 10*0.25-2 = 0.5 >= 0`), the low-value guest's `net = 1*0.25-2 < 0`
stays out — revenue `2`. Pricing to keep the lane to a single scarce buyer beats pricing for
"everyone who seemed willing".
