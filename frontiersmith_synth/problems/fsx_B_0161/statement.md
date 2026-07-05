# Green-Wave Phasing: Coordinating a Traffic-Signal Grid

## Story

A transportation authority controls an `N x N` grid of traffic signals. Every signal
runs on the **same fixed cycle time**, so the only thing you may choose is each
signal's **offset** (its green phase, measured as an angle `theta` in radians). Set the
offsets well and platoons of vehicles reinforce each other into coherent "green waves";
set them badly and the flow scatters.

The authority models the aggregate flow with a fixed, seeded wave simulator. Treating
each intersection as a unit-strength oscillator with complex amplitude
`exp(i * theta[r][c])`, the simulator computes the grid's **flow spectrum** as the
centered 2-D discrete Fourier transform of that field, and reads off the flow
**intensity** at every point of the spectrum as the squared magnitude. Physical energy
is conserved: the total intensity over the whole spectrum is a constant that does not
depend on your offsets -- you can only **redistribute** flow, not create it.

The city has designated a set of **target corridors**: `M` specific points in the flow
spectrum where sustained, balanced throughput is wanted (an off-center "spot array").
You must choose the offsets so that flow concentrates on those corridors *and* is
shared evenly among them. This is exactly the diffraction-efficiency-plus-uniformity
tension of designing a phase mask that lights up a target spot array: pouring energy
into the corridors is easy, but a phase-only device cannot make them all perfectly
bright *and* perfectly equal -- there is always residual scatter, so the best
achievable score stays below the ideal.

## Input (public instance, one JSON object on stdin)

```json
{
  "name": "grid11",
  "n": 8,
  "spots": [[u0, v0], [u1, v1], ...]
}
```

- `n` (int): the signal grid is `n x n`; the flow spectrum is also `n x n`, indexed
  after an `fftshift` so index `(n//2, n//2)` is the zero-frequency (DC) center.
- `spots` (list of `M` pairs `[u, v]`, `0 <= u, v < n`): the target corridors, i.e. the
  points of the flow spectrum you want bright and balanced. They are distinct and never
  include the DC center.

## Output (one JSON object on stdout)

```json
{"phases": [[t00, t01, ..., t0(n-1)], ..., [ ... row n-1 ... ]]}
```

- `phases` is an `n x n` array of real, finite floats: `phases[r][c]` is the offset
  `theta[r][c]` in radians. Any real value is allowed; the simulator uses it modulo
  `2*pi`.

Any of the following makes the instance score `0.0`: wrong shape (not `n x n`), a
non-numeric or non-finite entry (`NaN`/`Infinity`), a crash, a timeout, or output that
is not the JSON object above.

## Objective and scoring (deterministic)

The evaluator simulates the flow spectrum itself from your `phases`:

```
U          = exp(i * phases)                 # n x n complex field, unit amplitude
F          = fftshift(fft2(U))               # centered flow spectrum
I          = |F|^2                            # flow intensity
E_total    = sum(I)                           # constant (= n^4), independent of phases
efficiency = ( sum of I over the M target spots ) / E_total       # in [0, 1]
uniformity = ( min spot intensity ) / ( max spot intensity )      # in [0, 1], 0 if all dark
M_cand     = efficiency * uniformity                              # composite, in [0, 1]
```

The composite is a **product**: both factors must be high. Pouring lots of flow onto the
corridors but leaving one of them dark (low uniformity) tanks the score, and so does
spreading flow evenly but weakly (low efficiency). A phase-only grid cannot maximize both
at once, which is the core tension.

References (computed by the evaluator, never sent to you):

- `M_base` = composite of the **all-zero offset** design (every `theta = 0`). That design
  dumps all flow into the DC center, leaving every target corridor dark, so
  `M_base = 0`. It is the weak baseline.
- `M_ub = 1.0` = the ideal (all flow on the corridors, perfectly equal). A phase-only
  grid cannot reach it, so this is a loose, unreachable upper bound -> permanent headroom.

Normalized per instance with an affine anchor (weak baseline -> 0.1, ideal -> 1.0):

```
r = clamp( 0.1 + 0.9 * (M_cand - M_base) / (M_ub - M_base), 0, 1 )
```

Reproducing the all-zero design scores about `0.1`; concentrating balanced flow on the
corridors scores higher, capped below `1.0`. Your final score is the mean of `r` over
all instances -- a mix of grid sizes and corridor counts, including larger held-out
grids with more corridors, so a good strategy must generalize.

## Notes

- Scoring depends only on the `phases` you emit; it never measures wall-clock time.
  Treat the per-instance limit as an operation budget for iterative refinement
  (Gerchberg-Saxton / iterative-Fourier-transform, gradient ascent, grating
  superposition, random restarts, simulated annealing).
- Your program is run in an isolated subprocess and sees only the public instance above.
