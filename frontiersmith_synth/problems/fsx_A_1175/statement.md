# Missing Wedge: Reconstructing a Cross-Section from Too Few Angles

## Problem
A flat cross-section is modeled as an `N x N` grid of material indices, each pixel
holding one value from a small declared palette (e.g. `{0,1,2,3}`, void through dense
material). The true object is **piecewise-constant**: it is built from a handful of
large axis-aligned blocks, so most of the grid is locally flat.

A scanner records the object's discrete **Radon transform**: for each of `K` given
angles `theta` (degrees), every pixel's center `(x, y) = (i - (N-1)/2, j - (N-1)/2)` is
projected onto the direction `theta` and dropped into an integer bin
`b = round(x*cos(theta) + y*sin(theta)) + R//2` (clipped to `[0, R-1]`), where
`R = 2*ceil(N*sqrt(2)/2) + 1`. The sinogram row for that angle is the sum of pixel
values landing in each bin. **Only `K` angles are given, and they are often confined to
a narrow angular wedge** (e.g. `0..60` degrees) rather than spread over the full
`0..180` range — the classic **missing-wedge** setup: whole ranges of viewing direction
are simply never observed.

Your job: output an `N x N` grid of palette values that best explains the true,
unobserved object. Two forces matter. First, a reconstruction consistent with the given
sinogram is not unique when the wedge is missing — many grids explain the visible data
equally well, so blind inversion is underdetermined. Second, the object is known to be
piecewise-constant with only a few materials: a good reconstruction should look like a
small number of flat blocks, not textured noise. Filtered backprojection (direct
transform inversion) ignores this second fact entirely, so on a severe missing wedge it
produces smeared, streaky artifacts precisely in the unobserved directions. Treating
reconstruction instead as a constrained-optimization problem — fit the visible data
while actively preferring flat, blocky structure — recovers what inversion alone cannot.

## Input (stdin)
```
N test_id R
P
p_0 p_1 ... p_{P-1}
K
theta_0 theta_1 ... theta_{K-1}
row_0 (R integers)
...
row_{K-1} (R integers)
```
`test_id` is informational only. `p_0 < p_1 < ... < p_{P-1}` are the palette values
(integers). `theta_k` are integer degrees in `[0, 180)`. `row_k` is the length-`R`
sinogram for angle `theta_k`, computed exactly as described above.

## Output (stdout)
`N` lines, each with `N` space-separated values from the palette — pixel `(i, j)` is
row `i`, column `j`.

## Feasibility
Output is valid iff **all** hold: exactly `N` lines, each with exactly `N` tokens; every
token is finite and equals one of the declared palette values (tolerance `1e-6`); no
extra or missing tokens. Any violation scores `Ratio: 0.0`.

## Objective (maximize)
The checker knows the true object that generated the sinogram (it is not shown to you)
and scores your grid by two ingredients:
- **Structural accuracy**: the fraction of pixels whose value is within one palette
  step of the true pixel there.
- **Held-out projection consistency**: your grid is re-projected (same Radon rule
  above) at several angles you were *not* given, and compared to the true object's
  projections there.
These combine into a single quality score `F` (a fixed weighted blend, weights not
disclosed). The checker also builds its own naive single-flat-fill baseline `B` (best
constant palette value matching the overall mean density) and reports
`Ratio = min(1, F / B / 10)`, so the naive baseline lands near `0.1` and genuine
reconstructions score higher. A submission that only fits the visible wedge but ignores
the object's blocky structure will score well below one that also gets the flat
regions right.

## Constraints
`10 <= N <= 20`, `4 <= K <= 12`, `P = 4`, time limit 5s, memory 512MB.

## Example (illustrative only)
For a tiny `4x4`, 1-material object, a submission matching every visible pixel and
every held-out projection would score at the top of the achievable range; a flat guess
scores near `0.1`; a grid violating the palette (e.g. printing `7` when `P=4` covers
`{0,1,2,3}`) scores `0.0`.
