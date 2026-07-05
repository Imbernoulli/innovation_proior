# Tweezer-Array Wiring: Phase-Mask Hologram for a Qubit Register

A neutral-atom quantum lab "wires up" its qubit register by carving a single laser
beam into an **array of optical tweezers** — bright focal spots, one trap per qubit.
The beam passes through a phase-only spatial light modulator (SLM), which imprints a
phase mask `phi[y,x]` on the fixed illumination field; a lens then Fourier-transforms
the modulated field to the trapping plane.

Your job: design the phase mask so the trapping plane has bright, **equally bright**
spots exactly at the target qubit sites — high diffraction **efficiency** (light lands
in the traps, not wasted) and high **uniformity** (every qubit sees the same trap
depth). This is a pure-numpy, hardware-free, deterministically scored simulation.

## Physics (fixed, deterministic)

Given the (real, non-negative) illumination amplitude `A[y,x]` — a truncated Gaussian
times a fixed seeded ripple modelling the real, non-ideal beam — the SLM applies phase
only and the lens Fourier-transforms:

```
U_in  = A * exp(1j * phi)
U_out = numpy.fft.fftshift( numpy.fft.fft2( U_in ) )
I     = numpy.abs(U_out) ** 2
```

Target qubit sites are pixel coordinates `(r, c)` indexing the `M x M` array `U_out`.

## Metrics (higher is better)

```
efficiency  eta = ( sum of I over the target pixels ) / ( sum of I over ALL pixels )
uniformity  u   = 1 - (Imax - Imin) / (Imax + Imin)     over the target-pixel intensities
composite   Q   = eta * u
```

## Candidate program (isolated stdin -> stdout)

Read ONE JSON public instance from stdin; write ONE JSON answer to stdout.

**Public instance**
```json
{
  "name": "reg_b",
  "M": 48,
  "seed": 102,
  "spots": [[r, c], ...],
  "amp":  [[...], ...]
}
```
- `M` — grid side length.
- `spots` — `K` target pixel sites, each `0 <= r,c < M`.
- `amp` — the `M x M` illumination amplitude `A` (>= 0). Use it: the beam is non-uniform.

**Answer**
```json
{ "phase": [[...], ...] }
```
- `phase` — an `M x M` array of finite real numbers (radians; any finite value; only
  `exp(1j*phase)` matters, so scale/offset are free).

A mask is **valid** iff `phase` is an `M x M` array of finite reals. Wrong shape, a
non-finite entry (nan/inf), a crash, a timeout, or non-JSON scores **0** on that instance.

## Scoring

For each instance the evaluator computes `Q_base`, the composite of a weak analytic
baseline (plain in-phase superposition of one grating per trap), and your `Q_cand`, then
normalizes with an affine anchor (weak baseline -> 0.1, physically-perfect Q=1 -> 1.0):

```
r = clamp( 0.1 + 0.9 * (Q_cand - Q_base) / (1.0 - Q_base), 0, 1 )
```

Reproducing plain superposition scores ~0.1. The ideal `Q = 1` is unreachable for a
phase-only element under a truncated, rippled Gaussian beam, so even strong optimizers
keep headroom below 1.0. The reported score is the mean of `r` over all instances
(including harder, larger held-out registers). **Objective: maximize.**

## Strategies (open-ended)

- Plain superposition of gratings (the weak baseline).
- Random-phase superposition — de-correlate the traps to boost efficiency.
- Weighted Gerchberg-Saxton — feedback loop that equalizes trap depths.
- Seeded direct search / simulated annealing on the phase pixels; adaptive-additive or
  mixed-region-amplitude-freedom variants that trade uniformity for extra efficiency.

There is no easy optimum: efficiency and uniformity trade off, the beam is non-ideal,
and the phase-only constraint bounds what is reachable.
