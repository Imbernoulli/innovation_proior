# Data-Center Overhead Cooling-Gantry Sizing

A hyperscale data hall carries its liquid-cooling plant -- coolant distribution units,
chilled-water risers and manifold headers -- on a long **overhead steel gantry** that
spans the aisle between two structural columns. The gantry is modelled as a
**statically-determinate 2D pin-jointed truss** (a triangulated Pratt-style beam): joints
connected by straight bar members, **pinned** at one column and on a **roller** at the
other. The suspended cooling hardware hangs from the bottom-chord joints (downward point
loads) and hot-aisle airflow / seismic bracing pushes laterally on the top chord.

You must choose a **cross-sectional area for every bar** so that the gantry is as **light**
(cheap) as possible while remaining safe. This is a *feasibility-gated* objective: a design
that violates **any** limit is worthless.

Because the truss is statically determinate, the member **axial forces depend only on the
geometry and loads, not on your chosen areas**; the joint **displacements** do depend on the
areas (thicker bars deflect less).

## Model (linear-elastic truss FEM)

Given joint coordinates, bar connectivity, joint loads, supports, Young's modulus `E`, and
your chosen areas, the evaluator assembles the global stiffness matrix, solves `K u = F`
for the joint displacements `u`, and computes each bar's axial stress
`stress_e = E * elongation_e / length_e` (positive = tension, negative = compression).

## Hard constraints (all must hold, else the instance scores 0)

1. **Yield gate:** `|stress_e| <= sigma` for every bar `e`.
2. **Buckling gate:** every **compression** bar (`stress_e < 0`) must satisfy
   `|stress_e| <= sigma_cr_e`, where the Euler critical stress is
   ```
   sigma_cr_e = pi^2 * E * kappa * area_e / length_e^2
   ```
   (section family with second moment `I = kappa * area^2`, so the critical stress **grows
   with the chosen area** -- long, slender, lightly-thickened compression members buckle
   first).
3. **Sag gate:** every joint displacement magnitude `sqrt(ux^2 + uy^2) <= disp_limit`.
4. **Bounds:** `a_min <= area_e <= a_max` for every bar.

## Objective (minimize)

Total steel weight (density taken as 1):

```
weight(areas) = sum_e  area_e * length_e
```

## Candidate contract (isolated program)

Your program reads ONE JSON object (the public instance) from **stdin** and writes ONE JSON
object to **stdout**. It runs in an isolated subprocess: it sees only the public instance
below and nothing of the evaluator.

### Public instance (stdin) schema
```json
{
  "nodes":  [[x, y], ...],        // joint coordinates
  "bars":   [[i, j], ...],        // each bar connects joint i and joint j (M bars)
  "loads":  [[fx, fy], ...],      // external force at each joint
  "fixed":  [[bx, by], ...],      // per joint: is x-DOF / y-DOF fixed (support), 0/1
  "E":      <float>,              // Young's modulus
  "sigma":  <float>,              // allowable stress magnitude (yield)
  "disp_limit": <float>,          // allowable joint displacement magnitude (sag gate)
  "a_min":  <float>,              // min cross-sectional area
  "a_max":  <float>,              // max cross-sectional area
  "kappa":  <float>               // section constant: sigma_cr = pi^2*E*kappa*area/length^2
}
```

### Answer (stdout) schema
```json
{ "areas": [a_0, a_1, ..., a_{M-1}] }   // one area per bar, in bar order, each in [a_min, a_max]
```

## Scoring

Let `b` be the weight of the uniform `a_max` design (heaviest; always feasible). For a
**feasible** answer with objective `obj = weight(areas)`:

```
r = min(1.0, 0.1 * b / obj)
```

The uniform `a_max` design scores exactly `0.1`; a design `k` times lighter than that
baseline scores `min(1, 0.1 * k)`. A malformed answer, an out-of-bounds or non-finite area,
or any constraint violation scores `0.0` on that instance. The reported score is the mean of
`r` over a fixed, seeded set of 10 instances (a mix of spans and per-instance section
slenderness `kappa` and sag-gate tightness, including larger held-out gantries).

## Notes / strategy

Sizing purely to the yield gate (fully-stressed design, `area_e = |force_e| / sigma`) is
light and always passes the yield gate, but it leaves slender compression members that
**buckle**, and on tight-clearance instances the thin gantry **sags** past `disp_limit` --
either rejects the whole design. A robust solver must respect **all three** physical gates
at once -- e.g. floor each bar to `max(yield-area, buckling-area)`, then stiffen (scale
areas up) until the sag gate is met. There is no single closed-form optimum, and the right
trade-off varies per instance.
