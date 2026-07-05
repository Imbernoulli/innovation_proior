# Regolith-Ablation Beam Splitter — Phase-Mask Design for a Mining Spot Array (Format B, isolated)

An autonomous asteroid-mining rig ablates regolith with a single high-power laser. To
drill many pits at once it puts a **diffractive optical element (DOE)** — a pixelated
**phase mask** — in front of the beam. A phase-only mask multiplies the uniform incident
field by `exp(i·phi[y][x])`; the rig's optics then form the **Fraunhofer far field**
(a 2-D FFT) onto the asteroid surface, where the light concentrates into an array of
bright spots. Your job is to design the phase mask so the laser is split into a
**prescribed array of `K` ablation spots** as *efficiently* and as *uniformly* as
possible.

You are given, per instance, the mask resolution `N` (an `N×N` pixel grid) and the list
of `K` target spot locations in the far-field grid (integer pixel coordinates). You output
one real phase value per pixel. The grader simulates propagation with a **fixed,
deterministic FFT** and returns a composite efficiency×uniformity score. There is no
closed form: aiming a plain grating at each spot causes the spots to interfere (some go
dark), and pure efficiency ignores uniformity — the two objectives trade off and the
mask must be refined iteratively.

## Public instance (stdin JSON)
```json
{
  "N": 24,                      // mask is N x N pixels; far field is also N x N
  "targets": [[ky, kx], ...],   // K desired spot locations (row ky, col kx), 0 <= . < N
  "seed": 3701,                 // provided for reproducible randomized strategies (optional to use)
  "pattern": "grid3x3"          // human-readable tag for the spot layout (informational)
}
```
`K = len(targets)`. No target sits on the DC pixel `(0,0)`.

## Answer (stdout JSON)
```json
{"phase": [[p00, p01, ..., p0,N-1], ..., [pN-1,0, ...]]}   // N rows x N cols of real radians
```
Exactly `N` rows, each of exactly `N` finite real numbers (any magnitude — phases wrap
mod 2π). Any other shape/type, or a non-finite (`nan`/`inf`) value, is rejected and
scores **0**.

## What the grader computes (the simulator)
Let `A = exp(i·phi)` (unit-amplitude, phase-only aperture). The far field is
`G = FFT2(A)` and the intensity is `I = |G|²`. With total power `P = Σ I` and the target
spot intensities `I_k = I[ky_k][kx_k]`:

- **Diffraction efficiency** `eta = (Σ_k I_k) / P` — fraction of laser power delivered
  into the wanted spots (light scattered elsewhere, including the undiffracted 0th order,
  is wasted).
- **Uniformity** `U = 1 − (I_max − I_min)/(I_max + I_min)` over the `K` spot intensities —
  `U = 1` when every ablation spot is equally bright, `U → 0` when one spot hogs the power
  and others go dark.
- **Composite metric** `M = eta · U` (higher is better). A mask that is efficient but
  lopsided, or uniform but leaky, both score poorly.

The grader runs this FFT itself on your `phase`; you only ever receive the public instance
above.

## Objective & scoring
**Maximize** `M`. Per instance the raw metric is normalized against a reference mask that
the grader builds itself — the classic **superposition-of-gratings** DOE
`phi_ref = angle( Σ_k exp(i·2π(ky_k·y + kx_k·x)/N) )`:

```
score_instance = min(1,  0.1 · M_yours / M_reference)
```

so simply reproducing the reference construction scores ≈ 0.1, and you must beat it to do
better. The final score is the mean over 8 fixed, seeded spot-array instances (grids,
rings, crosses, and asymmetric layouts of varying `K` and spacing). There is deliberate
headroom: even a strong iterative design does not saturate every instance.

## Suggested strategies (increasing sophistication)
- **Superposition of gratings** — sum a blazed grating per target and take the phase
  (the reference; suffers from inter-spot interference and dark spots).
- **Random-phase superposition / few-step refinement** — add per-grating phase offsets or
  a handful of Fourier back-and-forth steps to break the destructive interference.
- **Gerchberg–Saxton (iterative Fourier transform)** — repeatedly FFT the mask, replace
  the far-field amplitude on the target spots while keeping its phase, IFFT, and keep only
  the aperture phase; iterate to concentrate power uniformly into the spots.
- **Multi-restart GS / weighted (adaptive-additive) GS** — several random restarts and/or
  per-spot weight updates that push the dimmest spots up, keeping the best mask found.
