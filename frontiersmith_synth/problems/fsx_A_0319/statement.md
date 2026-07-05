# Cryostat Probe Layout: Minimum Star-Discrepancy Wiring

## Problem
A cryogenic quantum chip is calibrated by dropping `M` measurement probes onto the
normalised 2-D wafer square `[0,1]^2`. If the probe layout is biased, whole
sub-regions of the chip are under-sampled and the calibration is wrong. The
worst-case bias of a layout is *exactly* its **star discrepancy**: over every
axis-aligned corner box `[0,a) x [0,b)` (with `0 <= a,b <= 1`), the fraction of
probes inside the box should track the box area `a*b`.

Choose the positions of the `M` probes to make the layout as balanced as possible
(smallest star discrepancy).

## Input (stdin)
A single line:
```
d M
```
`d = 2` is the dimension and `M` is the number of probes to emit.

## Output (stdout)
Exactly `M` probe positions, one per line, each two real numbers:
```
x_1 y_1
x_2 y_2
...
x_M y_M
```
Each coordinate must satisfy `0 <= x,y <= 1`. Emit exactly `M` points (no more,
no fewer). Points may coincide.

## Feasibility
The output must contain exactly `2*M` finite floating-point tokens, every one in
`[0,1]`. Any missing/extra token, non-finite value (`nan`/`inf`), or out-of-range
coordinate makes the layout infeasible and scores 0.

## Objective (minimise)
Let `P` be the emitted point set. The score is driven by the exact 2-D star
discrepancy
```
D*(P) = sup over 0<=a,b<=1 of | #{ p in P : p_x < a and p_y < b } / M - a*b |.
```
The supremum is attained on the finite grid formed by the probe coordinates, so
the checker computes it **exactly** — no sampling, no randomness. **Smaller
`D*(P)` is better.**

## Scoring
The checker builds an internal baseline `B` = the star discrepancy of the trivial
"main diagonal" layout `p_i = ((i+0.5)/M, (i+0.5)/M)` (all probes on the diagonal,
leaving whole off-diagonal boxes empty). With `F = D*(P)`:
```
sc    = min(1000, 100 * B / max(1e-9, F))
Ratio = sc / 1000
```
Reproducing the diagonal layout scores about `0.1`; halving the discrepancy roughly
doubles the score; a `10x` lower discrepancy caps the ratio at `1.0`.

## Constraints
- `d = 2`, `16 <= M <= 128` across the difficulty ladder.
- Deterministic scoring: the star discrepancy is an exact geometric measure of the
  submitted point set only.

## Example (worked score)
Suppose `M = 4` and you emit the centred 2x2 grid
`(0.25,0.25) (0.25,0.75) (0.75,0.25) (0.75,0.75)`. Its exact star discrepancy is
`F = 0.4375` (e.g. the box `[0,0.75)x[0,0.75)` holds 1 of 4 probes, `1/4 - 0.5625
= -0.3125`; the tightest box gives `0.4375`). The diagonal baseline for `M = 4` is
`B = 0.53125`, so `sc = 100 * 0.53125 / 0.4375 = 121.4`, `Ratio = 0.121`. A
Hammersley layout of the same 4 points scores higher. (Numbers illustrative.)
