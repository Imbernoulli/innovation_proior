# Program the Leopard's Exact Spots

## Problem
A synthetic pelt is an `N x N` pixel grid split into an `R x R` array of square
**patches**, each `B = 48` pixels wide (`N = R*B`). Pigment forms by a deterministic
discrete reaction-diffusion process: every pixel holds a real activator level `x`,
initialised from a fixed pseudo-random seed (given in the input) and relaxed for
`T = 220` steps by

```
A(i,j)   = average of x over the rasterized disk of radius ra = 2 around pixel (i,j)
I(i,j)   = average of x over the rasterized ring [ra, kill_reach(i,j)] around pixel (i,j)
net(i,j) = feed * A(i,j) - kill_gain * I(i,j)
x(i,j)  += tanh(net(i,j)) - x(i,j)          [applied to every pixel, each step]
```
(a "rasterized disk of radius r" is every pixel of centre-distance `<= r+0.5`; a
"rasterized ring `[r_in,r_out]`" is every pixel of distance in `(r_in+0.5, r_out+0.5]`)
with fixed global gains `feed = 1.5`, `kill_gain = 0.7` (disk/ring averages wrap
around the torus). The only thing that varies **per patch** is `kill_reach`: how far
the inhibitory signal reaches. A short reach lets activator blobs sit close together
(fine spots); a long reach forces them apart (coarse spots). You may only pick
`kill_reach` from a published catalogue of 6 formulations:

| formulation | kill_reach (px) | calibrated wavelength (px) |
|---|---|---|
| 0 | 5  | 10.8 |
| 1 | 7  | 13.0 |
| 2 | 9  | 14.6 |
| 3 | 11 | 16.7 |
| 4 | 13 | 18.9 |
| 5 | 16 | 20.1 |

The calibrated wavelength column is the mean spot-to-spot spacing that formulation
produces when it covers the *whole* pelt in isolation. **A single formulation only
gives one texture.** Each patch, however, has its own *target* spacing it must
approximate -- so you must design a per-patch **field** of formulations.

## Input (stdin)
```
R seed
t[1][1] t[1][2] ... t[1][R]
...
t[R][1] t[R][2] ... t[R][R]
```
`R` is the patch-grid side, `seed` seeds the pixel initialisation, and `t[i][j]`
(an integer, `10 <= t[i][j] <= 21`) is patch `(i,j)`'s target wavelength in pixels.

## Output (stdout)
`R*R` integers (any whitespace layout), row-major: `f[1][1] ... f[1][R] f[2][1] ...`,
the chosen formulation index `f[i][j]` in `{0,...,5}` for every patch.

## Feasibility
Every printed token must be an integer in `[0,5]`, and there must be exactly `R*R`
of them. Any violation scores `0`.

## Objective & Scoring
The checker builds the full pixel-level `kill_reach` field from your choices (patch
`(i,j)` gets `kill_reach` of your formulation `f[i][j]`), runs the exact relaxation
above, then reads off each patch's *actual* spot wavelength by cropping its interior
32x32 pixels (an 8-pixel margin removed on each side to reduce contamination from
neighbouring patches) and locating the dominant radial frequency of a windowed 2D
FFT of that crop. Patch `(i,j)`'s match is

```
match(i,j) = clamp(1 - |ln(actual(i,j) / t[i][j])| / ln(1.55), 0, 1)
```

(0 if the crop's pixel-value standard deviation falls below a fixed floor -- i.e. the
patch is essentially blank -- in which case no frequency is read off at all). Your
objective is
`Match = average of match(i,j) over all R*R patches`. The checker also computes its
own baseline `Base`: the objective obtained by painting formulation 0 (the
catalogue's finest, shortest-reach formulation) over the *entire* pelt regardless of
the targets. The final score is

```
Ratio = min(1.0, 0.1 * Match / Base)
```

Reproducing the baseline scores `0.1`; ten times the baseline's match caps at `1.0`.
Neighbouring patches' pixel values physically mix during relaxation (real diffusion
across the boundary), so the *realised* wavelength near a patch's edge can drift
from the calibration table's isolated prediction -- the table is a strong guide,
not an exact oracle.

## Constraints
`3 <= R <= 6`, time limit 5 s, memory 512 MB. The checker's simulation is O(pixels)
per step and fully deterministic (single-threaded, seeded).

## Example
`R = 3`, `seed = 7`, targets a checkerboard `[[11,20,11],[20,11,20],[11,20,11]]`.
Formulations `f = [[0,5,0],[5,0,5],[0,5,0]]` (matching each patch to its own nearest
calibrated wavelength) read back actual wavelengths near
`[[11.3,22.6,11.3],[22.6,11.3,22.6],[8.9,22.6,10.1]]`, giving `Match ~= 0.78`. The
checker's own baseline (formulation 0 everywhere) only reaches `Base ~= 0.43` here
(it satisfies the low-target patches but misses the high-target ones badly), so
`Ratio = min(1.0, 0.1 * 0.78 / 0.43) ~= 0.18`. Painting any single global formulation
everywhere is structurally unable to satisfy both halves of a checkerboard at once --
exact numbers depend on the deterministic simulation.
