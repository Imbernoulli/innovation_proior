# Sentinel Grid: Minimum-Discrepancy Pandemic Contact-Net Surveillance

## Problem
A public-health agency runs a pandemic **contact-net surveillance** program. Every person in the
population is described by a point in a normalized feature cube `[0,1]^d`: the coordinates encode
strata such as normalized geographic longitude/latitude, age band, mobility index and workplace
density. To detect emerging transmission early, the agency must place `M` **sentinel probes**
(pop-up testing sites) at coordinates in this feature cube.

The single quality that matters is *coverage evenness*: for **every** axis-aligned sub-population
`[0, c_1) x ... x [0, c_d)` (a rectangular slice anchored at the origin corner of the cube), the
fraction of probes falling inside that slice should match the slice's volume as closely as
possible. If some rectangular sub-population holds far more (or far fewer) probes than its share,
that stratum is over- or under-surveilled and outbreaks there are caught late or resources are
wasted. This mismatch, maximized over all slices in the mean-square sense, is exactly the
(L2) **star discrepancy** of the probe set.

Your task: place the `M` probes so their star discrepancy is **as small as possible**.

This is genuinely open-ended. For almost every `(d, M)` the probe set of minimum star discrepancy
is **unknown**. Deterministic constructions (Kronecker/additive-recurrence lattices,
Hammersley/Halton sequences, shifted lattices) are strong but not optimal, and further
optimisation can beat any single construction.

## Input (stdin)
A single line with two integers:
```
d M
```
`d` is the feature-cube dimension and `M` is the number of sentinel probes to place.

## Output (stdout)
`M` lines, each with `d` space-separated real numbers in `[0,1]` — the coordinates of one probe.
Output exactly `M * d` numbers. All coordinates must be finite (no `nan`/`inf`) and lie in the
closed interval `[0,1]`. Probes may coincide; there is no minimum-separation requirement.

## Feasibility
The output is rejected (score `0`) unless it contains exactly `M * d` finite real numbers, each in
`[0,1]`. Extra tokens, missing tokens, non-numeric tokens, or any value outside `[0,1]` or
non-finite makes the submission infeasible.

## Objective (minimize)
Let `P = {x_1, ..., x_M}` be your probe coordinates. The **L2 star discrepancy** is
```
D2*(P) = sqrt( integral over c in [0,1]^d of ( (#{i : x_i < c}) / M - prod_k c_k )^2 dc ).
```
The checker evaluates this integral in **closed form** using Warnock's identity (exact, no
sampling):
```
D2*(P)^2 = 3^(-d)
          - (2^(1-d) / M) * sum_i  prod_k (1 - x_{i,k}^2)
          + (1 / M^2)     * sum_i sum_j prod_k (1 - max(x_{i,k}, x_{j,k})).
```
Smaller `D2*(P)` is better. The computation is exact and deterministic.

## Scoring
The checker builds an internal baseline `B` = the L2 star discrepancy of a deterministic
pseudo-random probe placement (a fixed linear-congruential scatter seeded only by `d` and `M`).
With `F = D2*(P)` your submission's discrepancy, the score is
```
sc    = min(1000, 100 * B / max(1e-9, F))
Ratio = sc / 1000
```
Reproducing the scatter baseline gives `Ratio ~= 0.1`. Halving the discrepancy relative to the
baseline roughly doubles the ratio; a set ten times better than the baseline caps at `Ratio = 1.0`.

## Constraints
- `d in {2, 3, 4, 5}`, `16 <= M <= 300`.
- Deterministic scoring; no randomness, wall-time or hardware enters the score.

## Example
For `d = 2, M = 4`, an illustrative (deliberately *bad*) placement puts all four probes near the
main diagonal, e.g. `(0.1,0.1), (0.4,0.4), (0.6,0.6), (0.9,0.9)`: the slice `[0,0.5)^2` has volume
`0.25` yet contains `2/4 = 0.5` of the probes, a coverage gap of `0.25`, so the discrepancy is
large. Spreading them anti-diagonally, e.g. `(0.1,0.6), (0.4,0.1), (0.6,0.9), (0.9,0.4)`, lowers the
discrepancy and raises the ratio. These coordinates are illustrative only, not a target.
