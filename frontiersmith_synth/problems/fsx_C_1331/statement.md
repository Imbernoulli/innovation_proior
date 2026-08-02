# Printing Smaller Than the Light: Mask Pre-Distortion

## Problem
You design a photomask for a lithography step. Light passes through the mask's
clear openings and exposes photoresist on a wafer; the resist prints (clears)
wherever the accumulated light **intensity** clears a dose threshold. The optics
blur the mask, so **the printed pattern is never the mask** — small features
shrink or vanish, tightly packed features merge, and convex corners round off
while concave corners fill in. This is the optical proximity effect.

The wafer is a grid of `N x N` pixels. You know the **target** pattern you want
printed (a 0/1 grid) and the exact, fixed optical model:

- **Blur kernel** (partial-coherence imaging, radius 2, separable):
  `A = [1, 2, 3, 2, 1]`, so `kernel(dx,dy) = A[dx+2] * A[dy+2]` for
  `dx,dy in {-2,...,2}` (cells outside the grid contribute 0).
- **Intensity** at pixel `(x,y)`: `I(x,y) = sum_{dx,dy} mask(x+dx,y+dy) * kernel(dx,dy)`,
  using your submitted **mask** (a 0/1 grid, same size as target).
- **Dose latitude**: the process runs at THREE fixed doses,
  `T in {33, 41, 49}` (out of a max possible intensity of 81). A pixel PRINTS
  at dose `T` iff `I(x,y) >= T`. Your mask must print well across all three,
  not just one — real fabs never hold dose perfectly at the nominal value.

You output a mask. It does **not** have to look like the target — in fact,
because of the blur, drawing the mask as the exact target ("the obvious
approach") reliably prints the wrong shape at every dose. The insight is to
**pre-distort the mask** against the known model so that the *printed* result,
after blurring, matches the target.

## Input (stdin)
```
N
<N lines, each an N-character string over {0,1}: the TARGET pattern>
```
`13 <= N <= 25`.

## Output (stdout)
```
<N lines, each an N-character string over {0,1}: your MASK>
```
Print exactly `N` lines, each of exactly `N` characters, each `0` or `1`.

## Feasibility
The output is valid iff **all** hold: exactly `N` lines follow (no extra
non-blank lines), each line has length exactly `N`, and every character is
`0` or `1`. Any violation scores `Ratio: 0.0`.

## Objective
For a mask, compute the printed pattern at each dose `T in {33,41,49}` and its
Jaccard overlap with the target: `IoU_T = |printed_T & target| / |printed_T | target|`.
Maximize the **process-window fidelity**:
```
F = 0.4 * mean(IoU_33, IoU_41, IoU_49) + 0.6 * min(IoU_33, IoU_41, IoU_49)
```
The `0.6` weight on the *worst* dose means a mask tuned to look good only on
average, or only at one dose, scores poorly — you must survive the whole
process window.

## Scoring
Let `B` be `F` computed for the baseline mask **equal to the target itself**
(the "obvious" choice — always positive here, since every planted feature
clears at least one dose under identity printing). With maximization
normalization:
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Reproducing the target as your mask scores exactly `Ratio = 0.1`.

## Constraints
- `13 <= N <= 25`. Time limit 5s, memory 512m.
- Some cases contain isolated sub-resolution features (shrink drastically,
  sometimes almost to nothing, at the highest dose under identity printing),
  tightly pitched dense features (bridge together at low doses and can
  vanish at high doses under identity printing), and mixed instances
  combining both scales in one target — a single uniform size correction
  cannot fix all of these at once.

## Example
Target: a single isolated `3x3` solid block in a `9x9` grid (rows/cols 3..5).
Printing the target itself gives per-dose `IoU = (1.0, 0.556, 0.111)`, so
`B = 0.4*0.556 + 0.6*0.111 = 0.289`. A mask that pre-distorts the block into a
larger `5x5` block (rows/cols 2..6) gives per-dose
`IoU = (0.36, 0.429, 0.692)` (mean `0.494`, min `0.36`), so
`F = 0.4*0.494 + 0.6*0.36 = 0.413`,
`sc = 100*0.413/0.289 = 142.9`, `Ratio = 0.1429` — a mask that "looks wrong"
(too big) scores better because it accounts for the blur.
