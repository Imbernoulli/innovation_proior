# Coating the Lens: Worst-Case Anti-Reflection Stack Design

## Problem
A lens is coated with up to `N_max` thin dielectric layers to cancel reflected light at a
design wavelength `lambda0`. You are given `K` candidate coating materials, each with a
refractive index `n_i` and a **deposition-thickness tolerance** `tol_i` (the fractional
thickness error a manufacturing run of that material typically incurs -- e.g. `tol_i=0.05`
means any layer built from material `i` may come out up to 5% thinner or thicker than
designed). You choose a stack of `L` (`1 <= L <= N_max`) layers -- for each layer, a
material and a nominal thickness in nanometres -- built between air (`n0 = 1`) and a glass
substrate of index `n_sub`.

Reflectance through a stack follows classical thin-film interference: at normal incidence a
single quarter-wave layer of index `n1 = sqrt(n0*n_sub)` cancels reflection almost exactly,
but each layer's phase `delta_j = 2*pi*n_j*d_j*cos(theta_j)/lambda0` depends on the *actual*
built thickness `d_j` and on the ray angle `theta_j` (linked to the incidence angle `theta0`
by Snell's law `n0*sin(theta0) = n_j*sin(theta_j)`). A stack tuned to a razor-sharp zero at
exactly `theta0 = 0` and exactly its nominal thickness can drift badly the moment the
manufactured thickness is off by its tolerance, or light arrives off-axis -- exactly what
real coatings face.

Your lens must work over an **angle range** `[0, theta_max]` (degrees) and survive
**independent per-layer thickness drift** of up to each material's own tolerance. Your score
is your stack's **worst-case reflectance** over this whole angle/tolerance space -- not its
value at the single nominal design point.

## Input (stdin)
```
N_max K
n_1 tol_1
...
n_K tol_K
n0 n_sub lambda0
theta_max_deg
```
`n0` is always `1.0`. Every `n_i, n_sub >= 1.3`. Thicknesses you output must lie in
`(0, 1200]` nanometres (a fixed constant, not given in the input).

## Output (stdout)
```
L
mat_1 d_1
...
mat_L d_L
```
`1 <= L <= N_max`; each `mat_j` is a 1-based index into the `K` materials; each `d_j` is the
nominal (as-designed) thickness in nm of layer `j`, ordered from the air-facing surface
(layer 1) down to the substrate (layer `L`).

## Feasibility
Rejected (score `0`) if: the token count does not match `1 + 2*L`; `L` is not an integer in
`[1, N_max]`; any `mat_j` is not an integer in `[1, K]`; any `d_j` does not parse as a finite
number in `(0, 1200]`.

## Objective and Scoring
The checker evaluates your stack's reflectance by the exact transfer-matrix method
(s-polarised, lossless dielectrics) over a fixed grid: 5 angles evenly spaced across
`[0, theta_max]`, and for each layer independently one of `{-tol_j, 0, +tol_j}` applied
multiplicatively to its thickness (**all** `3^L` combinations are checked, not just the
extremes). Let `worst_R` be the maximum reflectance found anywhere on that grid. Define the
suppression score `F = -10*log10(max(worst_R, 1e-6))` (decibels of worst-case suppression --
higher is better). The checker also builds an internal baseline `B`: a single layer of the
FIRST provided material (`mat_1`, no search for a better match), at its own nominal
normal-incidence quarter-wave thickness, scored by the exact SAME worst-case procedure. Then
```
sc    = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000
```
so reproducing the baseline scores `0.1`; a stack with meaningfully better worst-case
suppression scores higher, up to the `1.0` cap.

## Constraints
`2 <= N_max <= 5`, `2 <= K <= 5`, `0 <= theta_max <= 65`, `0.005 <= tol_i <= 0.11`,
`500 <= lambda0 <= 620`, `1.45 <= n_sub <= 1.9`, `1.3 <= n_i <= 2.4`.

## Example
Suppose `N_max=2, K=2`, materials `n_1=2.35 (tol 0.070)`, `n_2=1.38 (tol 0.005)`,
`n0=1, n_sub=1.52, lambda0=550, theta_max=0`. Baseline `B` always uses material 1
(`n=2.35`) at `550/(4*2.35)=58.5` nm, theta0=0, no perturbation (theta_max=0 here).
Output `1` / `1 58.5` reproduces it: `Ratio ~= 0.1`. Material 2 is far closer to the ideal
single-layer index `sqrt(1.52)=1.233` and has smaller tolerance, so `1 / 2 99.6` (roughly
`550/(4*1.38)`) already beats the baseline. (Illustrative FORM only -- most cases give a
wide `theta_max` and larger tolerances, where ignoring them scores far worse than
optimizing the true worst case.)
