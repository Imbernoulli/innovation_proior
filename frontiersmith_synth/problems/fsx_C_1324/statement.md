# Letting One Molecule Through and Not Its Twin: Membrane Selectivity Design

## Problem

You are designing a separation membrane that must let a **target** molecule
`T` permeate while blocking a chemically-similar **twin** competitor `C`
that is close in kinetic size. You control two independent design channels:

1. **Pore-size distribution.** Up to `K_max` pore *families*, each a radius
   `r_k` and an area fraction `w_k` (the `w_k` form a distribution: they are
   non-negative and sum to 1). A narrower pore lets less of everything
   through but discriminates more by size.
2. **Chemical functionalization (the solubility channel).** A single loading
   level `alpha in [0, alpha_max]` that coats every pore wall. Coating
   shrinks each pore's *effective* radius by `delta_coat * alpha`, but it
   also changes how soluble each species is in the membrane material,
   *independently of size* -- proportionally more for `T`, less for `C`.

For a pore family with radius `r_k`, effective radius `r_eff = r_k -
delta_coat*alpha`, and species `s` (`T` or `C`) with kinetic diameter `d_s`,
the steric passage fraction is
```
lam = d_s / (2 * r_eff)
D(lam) = 1 / (1 + exp(BETA * (lam - 1)))          BETA = 4.0 (fixed)
```
`D` is smooth and monotone: close to 1 when the pore is much bigger than the
solute, close to (but never exactly) 0 when the pore is much smaller. This
is the fundamental **permeability-selectivity trade-off**: shrinking a pore
always buys some size-selectivity but never for free.

Solubility depends only on chemistry, not geometry:
```
Sol_s(alpha) = base_sol_s * (1 + alpha * chi_s)
```
`chi_T > 0` (the target likes the functional group more as loading grows),
`chi_C < 0` (the twin likes it less).

Permeability of species `s` through the whole membrane:
```
P_s = sum_k  w_k * D(lam_{s,k}) * Sol_s(alpha)
```
The **separation factor** is `S = P_T / P_C`. The design must also deliver a
minimum **throughput** `P_min` of the target; realized quality is
discounted proportionally if you fall short:
```
F = S * min(1, P_T / P_min)
```

## Input (stdin)

One line, 12 whitespace-separated numbers:
```
d_T d_C chi_T chi_C base_sol_T base_sol_C K_max r_min r_max alpha_max delta_coat P_min
```
`d_T, d_C`: kinetic diameters. `chi_T, chi_C`: affinity coefficients.
`base_sol_T, base_sol_C`: baseline solubilities. `K_max`: max pore families
you may use. `r_min, r_max`: allowed pore-radius range. `alpha_max`:
max functionalization loading. `delta_coat`: coating shrinkage per unit
alpha. `P_min`: required target throughput.

## Output (stdout)

```
K
alpha
r_1 w_1
r_2 w_2
...
r_K w_K
```
`K` (integer, `1 <= K <= K_max`) pore families, then `alpha`, then `K` lines
each with a radius and its area fraction.

## Feasibility

Rejected (score `0`) if: `K` out of `[1, K_max]`; `alpha` non-finite or
outside `[0, alpha_max]`; any `r_k` non-finite or outside `[r_min, r_max]`;
any `w_k` negative or non-finite; or the `w_k` do not sum to `1` (tolerance
`1e-6`).

## Objective

**Maximize `F`** as defined above.

## Scoring

Let `B` be `F` for the checker's own baseline construction: one pore family
at `r_max`, `alpha = 0` (wide open, no chemistry). Reported score:
```
Ratio = min(1000, 100 * F / B) / 1000
```
Reproducing the baseline scores `0.1`; a `10x` improvement in `F` caps the
ratio at `1.0`.

## Constraints

Deterministic scoring; each instance's randomness is seeded by the test id.
`d_T, d_C` are on the order of `0.3-0.7`; `r_min < r_max` are on a
comparable scale; `K_max` between 2 and 6; time limit 5s.

## Example

Suppose (illustrative numbers only) `d_T=0.40, d_C=0.55, chi_T=0.3,
chi_C=-0.3, base_sol_T=base_sol_C=1.0, K_max=2, r_min=0.12, r_max=1.2,
alpha_max=1.0, delta_coat=0.02, P_min=0.3`. The baseline (`r=1.2, alpha=0`)
gives `lam_T~0.167`, `D_T~0.9656`, `P_T~0.9656`; `lam_C~0.229`, `D_C~0.9562`,
`P_C~0.9562`; `S~1.010`, `thr=1` (comfortably above `P_min`), so `B~1.010`. A
design using
a smaller, chemistry-boosted pore can raise `S` well above `1` while still
clearing `P_min`, scoring well above `0.1`; a pore shrunk purely by size
with no chemistry help either wastes throughput margin or barely improves
`S` when `d_T` and `d_C` are close -- that is the trap this problem plants
on several of its ten test cases.
