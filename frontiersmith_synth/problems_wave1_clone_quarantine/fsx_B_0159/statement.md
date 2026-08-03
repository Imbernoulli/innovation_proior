# Vineyard Irrigation Trellis Sizing

A sloped vineyard is served by a long steel **trellis** that carries the drip
irrigation lines across a span between two ground anchors. The trellis is modelled
as a **statically-determinate 2D pin-jointed truss** (a triangulated beam): joints
connected by straight bar members. Water-filled drip lines hang from the joints
(downward loads) and a lateral wind gust pushes on the frame.

You must choose a **cross-sectional area for every bar** so that the trellis is as
**light** (cheap) as possible while remaining safe. This is a *feasibility-gated*
objective: a design that violates **any** limit is worthless.

## Model (linear-elastic truss FEM)

Given joint coordinates, bar connectivity, joint loads, supports, Young's modulus
`E`, and your chosen areas, the evaluator assembles the global stiffness matrix,
solves `K u = F` for the joint displacements `u`, and computes each bar's axial
stress `stress_e = E * elongation_e / length_e`.

## Hard constraints (all must hold, else the instance scores 0)

1. **Yield gate:** `|stress_e| <= sigma` for every bar `e`.
2. **Sag gate:** every joint's displacement magnitude `sqrt(ux^2 + uy^2) <= disp_limit`.
3. **Bounds:** `a_min <= area_e <= a_max` for every bar.

## Objective

Minimize total steel weight (density taken as 1):

```
weight(areas) = sum_e  area_e * length_e
```

## Candidate contract (isolated program)

Your program reads ONE JSON object (the public instance) from **stdin** and writes
ONE JSON object to **stdout**. It runs in an isolated subprocess: it sees only the
public instance below and nothing of the evaluator.

### Public instance (stdin) schema
```json
{
  "nodes":  [[x, y], ...],          // joint coordinates
  "bars":   [[i, j], ...],          // each bar connects joint i and joint j
  "loads":  [[fx, fy], ...],        // external force at each joint
  "fixed":  [[bx, by], ...],        // per joint: is x-DOF / y-DOF fixed (support)
  "E":      <float>,                // Young's modulus
  "sigma":  <float>,                // allowable stress magnitude
  "disp_limit": <float>,            // allowable joint displacement magnitude
  "a_min":  <float>,                // min cross-sectional area
  "a_max":  <float>                 // max cross-sectional area
}
```

### Answer (stdout) schema
```json
{ "areas": [a_0, a_1, ..., a_{M-1}] }   // one area per bar, in bar order, in [a_min, a_max]
```

## Scoring

Let `b` be the weight of the uniform `a_max` design (heaviest; always feasible).
For a **feasible** answer with objective `obj = weight(areas)`:

```
r = min(1.0, 0.1 * b / obj)
```

The uniform `a_max` design scores exactly `0.1`; a design `k` times lighter than
that baseline scores `min(1, 0.1 * k)`. A malformed answer, an out-of-bounds area,
a non-finite value, or any constraint violation scores `0.0` on that instance. The
reported score is the mean of `r` over a fixed, seeded set of instances (a mix of
small and larger, held-out spans, with per-instance sag-gate tightness).

## Notes / strategy

Sizing purely by stress (fully-stressed design, `area_e = |force_e| / sigma`) is
light and always passes the yield gate, but on instances with a tight `disp_limit`
the slimmer trellis sags past the sag gate and is rejected. A robust solver must
respect **both** gates -- e.g. size by stress, then stiffen (scale areas up) until
the sag gate is met. There is no single closed-form optimum, and the right trade-off
varies per instance.
