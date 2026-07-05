# Interstellar Relay Beam Fan-Out Phase Mask

An interstellar relay station must split a **single transmit laser** into a fan of
**downlink beams**, one aimed at each of several remote relay nodes. The only steering
element is a **phase-only spatial light modulator (SLM)**: an `N x N` grid of pixels,
each imposing a phase shift on a uniformly illuminated, unit-amplitude aperture. The
far field (Fraunhofer diffraction) is the 2-D FFT of the aperture field, and every
far-field pixel is one **diffraction order** -- one possible downlink direction.

You must choose the `N x N` phase mask so that, in the far field:

1. as much transmitted power as possible lands on the `M` requested spots
   (**diffraction efficiency**), and
2. that power is spread **evenly** across the spots (**uniformity**) -- an unevenly
   lit fan starves the dimmest relay node.

These goals compete: a single blazed grating is perfectly efficient into **one** spot
but ignores the rest. The score is a composite that rewards both.

## Forward model (exactly what the evaluator simulates)

Given your phase mask `phase` (an `N x N` real array, radians):

```
E    = exp(1j * phase)                 # uniform unit illumination, phase only
G    = fftshift(fft2(E))               # far field; DC (order 0,0) at pixel (N//2, N//2)
I    = |G|**2                          # far-field intensity
eff  = sum(I over the M target pixels) / sum(I over all N*N pixels)     # in [0, 1]
unif = min(spot_I) / max(spot_I)     # dimmest spot / brightest spot    # in [0, 1]
obj  = eff * unif                                                       # in [0, 1]
```

The uniformity term is the **harsh min/max ratio**: a single starved downlink node
(a dim spot) drags the whole score down, so you cannot ignore any target.

`fft2` / `fftshift` are the standard NumPy definitions. Target pixels are given in the
**fftshifted** convention, so pixel `(N//2, N//2)` is the zero-order (DC) direction and
targets are low-order pixels around it (never the DC pixel itself).

**Objective: maximize `obj`.**

## Candidate contract (isolated program)

Your program reads ONE JSON object (the public instance) from **stdin** and writes ONE
JSON object to **stdout**. It runs in an isolated subprocess: it sees only the public
instance below and nothing of the evaluator.

### Public instance (stdin) schema
```json
{
  "N": <int>,                       // grid size (mask is N x N, far field is N x N)
  "targets": [[r, c], ...]          // M requested far-field pixels (fftshifted coords)
}
```

### Answer (stdout) schema
```json
{ "phase": [[p00, p01, ...], ...] }  // N rows, each of length N; real radians
```

Any real phase value is allowed (only `exp(1j*phase)` is used). The answer is rejected
(score `0` on that instance) if it is not a dict, if `phase` is not an `N x N` nested
list, if any value is non-finite, or if the resulting composite `obj` is `<= 0`.

## Scoring

Let `b` be the composite of the evaluator's **binary phase-mask** baseline (the
cheapest fabricable DOE: threshold the grating superposition to two levels, `0` / `pi`).
For a valid answer with composite `obj`:

```
r = min(1.0, 0.1 * obj / max(b, 1e-12))
```

The cheap binary mask (`obj == b`) scores exactly `0.1`; a mask whose composite is `k`
times the baseline scores `min(1, 0.1*k)`. The reported score is the mean of `r`
over a fixed, seeded set of instances (small grids plus larger held-out grids, with
varying target counts and layouts).

## Notes / strategy

The binary baseline wastes power in conjugate ghost orders and lights the spots
unevenly. Even the one-shot analog superposition (keep the full continuous phase, not
just its sign) already beats it. Iterative phase retrieval -- **Gerchberg-Saxton** and its **weighted / adaptive-additive**
variants -- alternates between the aperture plane (enforce unit amplitude, keep phase)
and the far-field plane (enforce the target amplitudes on the spots, zero elsewhere),
and can additionally **reweight** the per-spot targets to equalize brightness. Other
approaches (Dammann-grating layouts, direct search, simulated annealing on the pixel
phases) are also viable. There is no closed-form optimum, and the best efficiency /
uniformity trade-off varies per instance.
