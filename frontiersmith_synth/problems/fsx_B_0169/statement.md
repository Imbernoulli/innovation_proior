# Festival Stage Layout — Binary Phase-Plate Spot-Array Beam Shaper

## Story

You are lighting a festival main stage. A single collimated laser projector
illuminates a flat **phase plate** (a "gobo" / binary DMD panel). The plate is
divided into an `N x N` grid of cells; each cell can be etched to one of only
`L` discrete optical phase levels (the panel is a **binary** device, `L = 2`,
so each cell is either `0` or `π`). The far-field image the plate throws onto
the backdrop is the Fraunhofer diffraction pattern of the illuminated aperture:

```
field  = fftshift( fft2( exp(1j * phase) ) )          # phase is the N x N mask
I      = |field|**2                                    # backdrop intensity, DC at (N//2, N//2)
```

Your job: choose the phase mask so that the light collapses into a chosen
constellation of **spot** pixels on the backdrop (the show's "beam array"),
with the spots **equally bright** and as little light wasted as possible.

Because the plate is binary, its far field is conjugate-symmetric: every spot
is mirrored by an unavoidable twin order carrying comparable power, so no design
can put more than ~half the light into an asymmetric target set. The art is in
balancing the spots against each other, not in beating physics.

## The candidate program (stdin → stdout)

Your program reads ONE JSON object (the public instance) from stdin and writes
ONE JSON object (your answer) to stdout.

```python
import sys, json
inst = json.load(sys.stdin)
# ... compute a phase mask ...
print(json.dumps({"phase": mask}))   # mask: N lists of N floats (radians)
```

### Public instance schema

```json
{
  "N": 64,                       // aperture grid is N x N cells
  "L": 2,                        // number of physical phase levels (binary)
  "unif_power": 3.0,             // exponent applied to the uniformity term (see scoring)
  "spots": [[r, c], ...],        // M target pixels in the centered N x N far field (DC at N//2)
  "seed": 137                    // seed defining the reference construction
}
```

### Answer schema

```json
{ "phase": [[float, ... N ...], ... N ...] }   // radians; anything finite is accepted
```

The device is physical: the grader **snaps every value to the nearest of the
`L` levels** (`round(phase / (2π/L)) * (2π/L)`) before propagating. You cannot
evade the binary constraint, so design with it in mind.

## Objective (MAXIMIZE)

Let `I` be the snapped mask's far-field intensity, and `I_k` the intensity at
the `M` target spots. Define

```
efficiency  eta = (sum_k I_k) / (sum of all I)
uniformity  U   = min_k I_k / max_k I_k
Q           = eta * U**unif_power
```

`Q` rewards throughput into the spots AND equal brightness across them.

## Scoring

For each instance the grader runs your program in an isolated subprocess,
validates the answer (shape `N x N`, all finite), computes `Q`, and normalizes
against a reference construction `Q_base` (a random-phase superposition of
blazed gratings, one per spot):

```
r = min(1.0, 0.1 * Q / Q_base)
```

A design equal to the reference scores ≈ 0.1; a well-balanced design scores
higher. The final score is the mean of `r` over all instances. Everything is
deterministic and seeded — the same mask always earns the same score.

## Hints

- A superposition of blazed gratings (one steering the beam to each spot, each
  with a random piston phase) is a solid starting point — it is the reference.
- Iterative Fourier-transform methods (Gerchberg–Saxton / IFTA) refine it:
  transform, force the target spots toward a common amplitude, keep the phase,
  transform back, keep the phase, repeat.
- Adaptive **per-spot weighting** (push harder on the dim spots) is what closes
  the uniformity gap on the denser boards. Re-snap to the device levels each
  iteration so the design stays robust to quantization.
