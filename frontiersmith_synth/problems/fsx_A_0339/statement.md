# Tide Pool Biodiversity Survey: Uniform Microhabitat Sampling

## Problem
A marine ecologist is surveying the microhabitats of a rocky-shore tide pool. Every
microhabitat is described by three normalized environmental gradients:

- **tidal elevation** (submerged ... exposed),
- **salinity** (brackish ... hypersaline),
- **substrate rugosity** (smooth rock ... deeply pitted).

Rescaled to `[0,1]`, the space of microhabitats is the unit cube `[0,1]^3`. You must
choose the locations of `M` survey quadrats so that the sample is spread as *uniformly*
as possible across the cube: every axis-aligned "corner" region should hold a fraction
of quadrats close to its volume. The uniformity of a placement is measured by its
**3D star discrepancy** (lower is better).

## Input (stdin)
One line with two integers:
```
M d
```
`M` is the number of quadrats to place; `d = 3` is the environmental dimension.

## Output (stdout)
Print exactly `M` quadrat locations, one per line (or whitespace-separated), each as
three real coordinates in `[0,1]`:
```
x_1 y_1 z_1
x_2 y_2 z_2
...
x_M y_M z_M
```
Exactly `3*M` numbers must be emitted. Coordinates must be finite and lie in
`[0,1]` (a tolerance of `1e-6` is allowed and clamped).

## Feasibility
- Exactly `3*M` finite numbers.
- Every coordinate in `[-1e-6, 1+1e-6]`.
Any violation (wrong count, `nan`/`inf`, out-of-range) scores `0`.

## Objective (minimize)
For a point set `P` of `N = M` points in `[0,1]^3`, the **star discrepancy** is
```
D*(P) = sup_{q in [0,1]^3} | (# points inside the box [0,q)) / N  -  vol([0,q)) |,
```
where `vol([0,q)) = q_x * q_y * q_z`. The checker computes `D*(P)` **exactly**: the
supremum is attained on the finite grid whose per-axis candidate coordinates are the
coordinates of the points together with `1.0`, evaluating both the closed count
(coords `<= q`, driving the over-full term) and the open count (coords `< q`, driving
the under-full term). Smaller `D*(P)` is better.

## Scoring
Let `F = D*(P)` be your placement's exact star discrepancy and let `B` be the star
discrepancy of the internal baseline — the `M`-point **space diagonal**
`((i+0.5)/M, (i+0.5)/M, (i+0.5)/M)`. The reported score is
```
Ratio = min(1000, 100 * B / max(1e-9, F)) / 1000.
```
Reproducing the diagonal gives `F = B` and `Ratio ~ 0.1`; halving the discrepancy
roughly doubles the score; a 10x improvement caps the score at `1.0`. There is no
known closed-form optimum for `M`-point 3D star discrepancy at these sizes, so many
strategies (Halton/Sobol sets, rank-1 lattices, jittered strata, local search) are
viable and the objective stays genuinely open-ended.

## Constraints
- `5 <= M <= 18` across the test ladder; `d = 3`.
- Deterministic scoring; exact grid evaluation with a fixed `1e-6` geometry tolerance.

## Example
For `M = 4`, the placement
```
0.125 0.375 0.700
0.625 0.875 0.100
0.375 0.125 0.500
0.875 0.625 0.900
```
spreads the four quadrats across the cube. The checker builds the induced anchor grid,
takes the largest gap between "fraction of quadrats inside a corner box" and "volume of
that box", and reports it as `F`; the diagonal baseline `B` is computed the same way,
and `Ratio = 100 * B / F / 1000` is printed on the final `Ratio:` line. (Illustrative
placement only — not an optimal set.)
