# Watchtower Beacon Panel — Spot-Array Diffraction Efficiency & Uniformity

## Setting (forest-fire watchtowers)
A forest-fire watchtower carries a **beacon panel**: an `N x N` grid of individually
tiltable mirror facets (a phase-only spatial light modulator). Collimated light
illuminates the panel with unit amplitude; each facet applies a phase shift
`phi[i][j]` (in radians). You must aim the beam so it simultaneously lights up a
ridgeline of `K` distant **lookout cells**.

The panel's far-field (Fraunhofer) illumination is simulated deterministically by a
discrete Fourier transform:

```
U = exp(1j * phi)                      # complex facet field, |U| = 1
G = fftshift( fft2(U) )                # far-field amplitude
I = |G|^2                              # far-field intensity, an N x N grid
```

Lookout cells are the far-field pixels listed in `targets` (row, col in `[0, N)`;
the undiffracted DC order at the center is never a target).

## Objective (MAXIMIZE)
Compute the composite beacon quality

```
eff = (sum of I over the K lookout cells) / (sum of I over all N*N pixels)
u   = min(I over lookouts) / max(I over lookouts)
metric = eff * u
```

`eff` rewards putting energy **on** the lookouts; `u` rewards splitting it **evenly**
so no lookout is left dim. Both lie in `[0, 1]`; `metric` in `[0, 1]`. A phase-only
panel cannot reach `metric = 1` (there is always stray light and residual imbalance),
so the frontier is open-ended: many strategies exist and the last few decibels of
uniformity are genuinely hard.

## Public instance (stdin, one JSON object)
```json
{"N": 24, "targets": [[r0, c0], [r1, c1], ...]}
```

## Answer (stdout, one JSON object)
```json
{"phase": [[phi_00, phi_01, ...], ...]}   // an N x N array of real radians
```
The array must be exactly `N x N`; every entry a finite real number. `nan`/`inf`,
wrong shape, or a missing `phase` key score 0 on that instance.

## Scoring
Per instance the evaluator computes your `metric` and normalizes against a fixed
reference construction `b` (an analytic superposition of steering gratings):

```
ratio = min(1.0, 0.1 * metric / b)
```

so the reference construction scores ~0.1 and there is headroom above it. The final
score is the mean ratio over 8 seeded instances (`K` ranging from 10 to 24). The
candidate runs in an isolated sandbox and sees only the public instance.

## Ideas
- Analytic **superposition of gratings** (one steering grating per lookout, keep the
  phase) — decent efficiency, uneven brightness.
- **Weighted Gerchberg-Saxton** phase retrieval: iterate FT / inverse-FT, re-imposing
  target amplitudes and re-weighting dim lookouts to equalize them.
- Track the best-uniformity iterate; anneal or direct-search the facet phases.
