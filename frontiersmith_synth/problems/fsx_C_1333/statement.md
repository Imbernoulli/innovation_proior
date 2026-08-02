# Not Just Less Water: Concrete Mix Design Under a Shrinkage Budget

*Strong, workable, and not cracking in a year.*

## Problem
You are proportioning one cubic meter of concrete. You choose an **aggregate blend**
(one of `K` graded sand/gravel options), a **cement content** `c` (kg/m3), a **water
content** `w` (kg/m3), and a **chemical admixture (superplasticizer) dosage** `p`
(a fraction of the cement mass, e.g. `p=0.02` means 2%).

Three effects are coupled:
- **Strength** rises as the water/cement ratio `wc = w/c` falls.
- **Workability** (the mix must be fluid enough to place and compact) requires enough
  water to lubricate the chosen aggregate blend. Each blend `j` has its own water
  demand `W0_j`: the water (kg/m3) needed, with *no* admixture, to reach the required
  slump. A better-graded blend packs tighter and needs less water. The admixture
  chemically lowers the water actually needed, by a saturating fraction
  `reduction(p) = wr_max * p / (p + p_half)`; the water you must supply is then
  `req_j(p) = W0_j * (1 - reduction(p))`.
- **Shrinkage-cracking risk** rises with `wc`, rises with total paste volume
  (cement+water), and falls with the aggregate's restraining volume; the admixture
  itself adds a small side-penalty (overdosing risks bleeding). Concretely, with
  `Vc = c/rho_c`, `Vw = w/rho_w`, `Vagg = 1 - air - Vc - Vw`:
  ```
  SCR = k1*wc + k2*(Vc+Vw) - k3*Vagg + k4*p
  ```
  must stay at or below a per-instance budget `risk_limit`.

Naively minimizing `wc` (max cement, min water) starves workability; naively adding
water back to fix workability can blow the shrinkage budget. The way out is choosing
the blend and dosage so workability is satisfied by *packing and chemistry*, not by
extra water -- freeing you to push cement as far as the shrinkage budget allows.

## Input (stdin)
```
K
rho_c rho_w air
c_min c_max
w_min w_max
wc_min wc_max
wr_max p_half p_max
k1 k2 k3 k4
A B
vagg_min risk_limit
W0_1
...
W0_K
```
`rho_c`, `rho_w` are densities (kg/m3); `air` is the fixed entrained-air volume
fraction. `W0_1..W0_K` are the per-blend water demands (kg/m3) described above.

## Output (stdout)
One line: `j c w p` -- the chosen blend index (`1..K`), cement content, water content
(both kg/m3), and admixture dosage (fraction of cement mass).

## Feasibility
All of the following must hold, else the output scores `Ratio: 0.0`:
- `1 <= j <= K`; `c,w,p` finite.
- `c_min <= c <= c_max`, `w_min <= w <= w_max`, `0 <= p <= p_max`.
- `wc_min <= w/c <= wc_max`.
- `w >= W0_j * (1 - wr_max*p/(p+p_half))` (workability met for the chosen blend/dosage).
- `Vagg = 1 - air - c/rho_c - w/rho_w >= vagg_min` (aggregate must remain the
  load-bearing skeleton -- not a paste-flooded mix).
- `SCR <= risk_limit` (as defined above).

## Objective
Maximize the 28-day strength surrogate `F = A - B*(w/c)`.

## Scoring
The checker builds its own reference recipe: blend `1`, no admixture, minimum cement
`c_min`, water exactly `W0_1` (this recipe is always feasible). Let `F_base` be its
strength. With `F` your feasible strength:
```
sc = min(1000.0, 100.0 * F / max(1e-9, F_base))
Ratio = sc / 1000.0
```
Matching the reference scores `0.1`; a mix reaching `10x` its strength margin caps at
`1.0`. Every violated constraint above scores `0.0`.

## Constraints
`3 <= K <= 6`. All numeric constants are given per test case in the input. Time limit
5s, memory 512MB.

## Example
Suppose (illustrative numbers, not a real test case) `K=2`, `W0 = [200, 150]`,
`c_min=290`, and the reference recipe gives `wc=200/290=0.690`, `F_base = A-B*0.690`.
A submission using blend `2` with dosage `p` that drops its required water to `120`,
paired with a larger cement content `c` still inside the shrinkage budget, achieves a
lower `wc` and hence a larger `F`, scoring above `0.1`.
