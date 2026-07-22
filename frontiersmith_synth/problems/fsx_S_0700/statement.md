# Two-Plane Phase Plate: One Etched Mask, Two Fresnel Focal Images

## Problem
You are etching a single **phase-only plate**: an `N x N` grid of pixels, each
imprinting a phase shift `theta[i][j]` (any real, in radians) on light passing
through it. The incident illumination has **fixed, uniform amplitude 1** at
every pixel — you control *only* the phase, never the amplitude, at the
aperture.

The same etched plate is used at **two different working distances**, modeled
as two independent discrete Fresnel-propagation operators (plane `A` and plane
`B`). Propagating through plane `X` (`X in {A,B}`) means: multiply the
aperture field by a fixed quadratic "chirp" phase curvature `alpha_X`, then
take a unitary 2D FFT to reach that plane's focal-plane field:

```
chirp_X[i,j]   = exp( i * alpha_X * ((i-c)^2 + (j-c)^2) ),  c = (N-1)/2
field[i,j]     = exp( i * theta[i,j] ) * chirp_X[i,j]        # |field|=1 everywhere
F_X            = FFT2(field, norm="ortho")                   # unitary -> energy-preserving
I_X            = |F_X|^2                                     # focal-plane intensity, N x N
```

Because the FFT is unitary and every aperture pixel has unit amplitude,
`sum(I_X) == N*N` **always**, independent of the phase mask — the plate cannot
create or destroy energy, it can only *redistribute* it. Each plane has its
own target intensity image, `T_A` and `T_B`. You must choose ONE mask
`theta` that makes `I_A` look like `T_A` *and* `I_B` look like `T_B`
*simultaneously* — a single phase mask, two independent constraints.

## Input (stdin)
```
N alpha_A alpha_B lambda_tv
T_A row 0            (N floats, >= 0)
...
T_A row N-1
T_B row 0
...
T_B row N-1
```

## Output (stdout)
```
N
theta row 0           (N floats, any finite real, radians)
...
theta row N-1
```
The first line must repeat `N`; each of the next `N` lines must have exactly
`N` finite tokens with `|theta[i][j]| <= 1e6`. Wrong shape, non-numeric,
`nan`/`inf`, or out-of-range tokens score `0` on that case.

## Objective (MINIMIZE)
For each plane, normalize both the achieved intensity and the target into
probability distributions (`P_X = I_X / (N*N)`, `Q_X = T_X / sum(T_X)`) and
measure a normalized mismatch:
```
nmse_X = sum((P_X - Q_X)^2) / sum(Q_X^2)
```
Also penalize a rough etch via the mean squared *wrapped* angular difference
between phase-adjacent pixels (horizontal + vertical neighbors), `tv(theta)`.
The full objective:
```
F(theta) = 0.5*(nmse_A + nmse_B) + lambda_tv * tv(theta)
```

## Scoring
The checker computes `F` for your mask and `B = F` for its own trivial
reference (the unmodulated flat plate, `theta = 0`), then
```
ratio = min(1.0, 0.1 * B / F)
```
so the flat plate scores ~0.1 and there is headroom for masks 10x better.
Mean ratio over 10 seeded instances is your score. A phase-only mask
generally **cannot** satisfy two independent full-image targets exactly, so
`ratio = 1.0` is not attainable — the frontier stays open.

## Constraints
`16 <= N <= 32`, `time limit 5s`, `memory 512MB`, each `.in <= 5MB`.

## Example (toy, illustrative arithmetic only — not real test data)
`N=2, alpha_A=0.3, alpha_B=-0.5, lambda_tv=0.1`, `T_A=[[1,0],[0,0]]`,
`T_B=[[0,1],[0,0]]`. The checker's own flat-mask baseline (`theta=0`) gets
`P_A=[[1,0],[0,0]]` (matches `T_A` exactly here) but `P_B=[[1,0],[0,0]]` too
(mismatches `T_B` badly) giving `B = 0.5*(0 + 2.0) = 1.0`.
A candidate mask `theta=[[0,pi],[0,0]]` yields `P_A=P_B=[[.25,.25],[.25,.25]]`,
`nmse_A=nmse_B=0.75`, `tv=4.935`, so
`F = 0.5*(0.75+0.75) + 0.1*4.935 = 1.2435`. Since `F > B` here this
particular candidate scores below the baseline
(`ratio = min(1,0.1*1.0/1.2435) = 0.0804`) — illustrating that an
uninformed phase jump can easily do *worse* than doing nothing; real
solutions must search for masks with `F < B`.

## Ideas
- Single inverse Fresnel transform of (a blend of) the targets, one plane only
  — fast, but never reconciles the fixed-amplitude aperture constraint or the
  second plane.
- **Two-plane Gerchberg-Saxton**: alternately propagate to plane A, impose
  `sqrt(T_A)` on the amplitude, propagate back, impose the fixed uniform
  aperture amplitude; repeat for plane B on the updated mask. Iterate.
- Multi-restart with best-iterate tracking; local/annealing search on `F`.
