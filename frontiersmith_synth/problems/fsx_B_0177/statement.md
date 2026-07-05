# Wildlife Corridor Guide-Light Phase Mask

A conservation team is building a **wildlife corridor** — a safe crossing that threads a
string of habitat patches past a highway. At night they gently draw nocturnal animals
along the corridor with a single low-power infrared **guide-light projector**. The
projector is a phase-only **spatial light modulator (SLM)**: an `N x N` grid of pixels,
each imposing a phase shift on a uniformly illuminated aperture. By Fraunhofer
diffraction the far field is the 2-D FFT of the aperture field, and each far-field pixel
is one diffraction order — one direction the beam can be aimed.

You must design the `N x N` phase mask so that, in the far field, the beam lands as a fan
of dim guide spots — **one spot per habitat patch, strung along the corridor** — that
reproduces a **prescribed illumination profile**: a large clearing (high weight `w_i`)
should get more light than a narrow underpass (low weight). Two goals fight each other:

* **Efficiency** — as much transmitted power as possible should land on the patches
  rather than leaking off-corridor.
* **Profile fidelity** — the achieved per-patch intensities should match the prescribed
  weights `w_i`, not just be equal or arbitrary.

A single blazed grating is perfectly efficient into ONE patch but ignores the profile, so
the objective is a composite that both terms must satisfy.

## Candidate contract

Your program reads ONE JSON object (the public instance) from **stdin** and writes ONE
JSON object (your answer) to **stdout**.

```python
import sys, json
inst = json.load(sys.stdin)
# ...compute the phase mask...
print(json.dumps({"phase": phase_2d_list}))
```

### Public instance schema
```json
{
  "N": 20,                                  // grid size; mask is N x N
  "targets": [[r, c], ...],                 // M patch pixels in the far field,
                                            //   indices into fftshift(fft2(.)) (DC at N//2,N//2)
  "weights": [w_1, ..., w_M]                // prescribed illumination weight per patch (>=1)
}
```
The patches lie (roughly) along a line — the corridor. The DC (central) pixel is never a
patch.

### Answer schema
```json
{ "phase": [[...N floats...], ... N rows ...] }   // the N x N phase mask, radians, finite
```
Any wrong shape, non-list, or non-finite entry scores **0** on that instance.

## Forward model (exactly what the evaluator computes)

```
E     = exp(1j * phase)                 # uniform unit illumination, phase-only
G     = fftshift(fft2(E))               # far field, DC at (N//2, N//2)
I     = |G|**2                          # far-field intensity
eff   = sum(I at patch pixels) / sum(I)                         in [0,1]
p_i   = I_i / sum_j I_j     (achieved fraction on patch i)
q_i   = w_i / sum_j w_j     (prescribed fraction on patch i)
fidel = 1 - 0.5 * sum_i |p_i - q_i|     # 1 - total-variation distance,  in [0,1]
obj   = eff * fidel                                             in [0,1]   (MAXIMIZE)
```

## Scoring

For each instance the evaluator computes a naive **baseline** `b`: the strawman single
blazed grating that dumps the whole beam onto the highest-weight patch (perfectly
efficient into that one patch, but blind to the rest of the corridor). Your per-instance
score is

```
r = min(1, 0.1 * obj / max(b, 1e-12))
```

so matching the naive baseline scores `0.1`; doing `k` times better scores
`min(1, 0.1*k)`. A malformed answer, wrong shape, a non-finite value, or `obj <= 0`
scores 0 on that instance. The final **Ratio** is the mean of `r` over all instances
(a mix of small and larger held-out grid sizes). Objective: **maximize**.

Scoring is deterministic and seeded. Your program is run in an **isolated subprocess**;
it sees only the public instance above.
