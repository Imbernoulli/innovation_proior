# Wind-Turbine Support Tower — Minimum-Mass Lattice Sizing (Format B, isolated)

A wind-farm turbine sits on top of a slender **2D pin-jointed steel lattice tower**.
The rotor pushes on the nacelle with a horizontal **thrust** (the wind load) and the
nacelle+rotor hang on the tower as a downward **dead weight**, both applied at the top
(hub height). The tower's geometry — node positions, which member connects which
nodes, the pinned base supports, and the applied loads — is fixed and given to you.
**Your job is to choose the cross-sectional area of every lattice member** so the whole
tower is as light as possible while still standing up under load.

A design is **feasible** only if it passes BOTH hard limits, checked by a linear-elastic
2D truss finite-element solve:

1. **Stress limit** — every member's axial stress satisfies `|σ_i| ≤ sigma`.
2. **Sway limit** — every nodal displacement magnitude satisfies `|u_d| ≤ u_max`
   (this is dominated by the horizontal top-of-tower sway).

An infeasible design (any limit exceeded), or an ill-formed answer, scores **0**. Among
feasible designs, lighter is better. There is no closed-form optimum you can copy: sizing
purely for stress can violate the sway limit, and thickening the whole tower for stiffness
wastes steel — the two constraints trade off. The tower is statically determinate, so
member forces do not change with your areas; that is exactly what lets a smart per-member
sizing beat a uniform one.

## Public instance (stdin JSON)
```json
{
  "nodes":  [[x, y], ...],            // node coordinates, meters (index = node id)
  "bars":   [[i, j], ...],            // each member connects node i to node j
  "fixed":  [dof, ...],               // clamped DOFs (dof = 2*node for x, 2*node+1 for y)
  "loads":  [[dof, value_N], ...],    // external nodal forces, Newtons
  "E":      2.0e11,                   // Young's modulus, Pa
  "rho":    7850.0,                   // density, kg/m^3
  "sigma":  2.5e8,                    // allowable |axial stress|, Pa
  "a_min":  1.0e-4,                   // min cross-sectional area, m^2
  "a_max":  2.0e-2,                   // max cross-sectional area, m^2
  "u_max":  <float>                   // allowable |nodal displacement|, m
}
```
The number of members is `m = len(bars)`.

## Answer (stdout JSON)
```json
{"areas": [a_1, a_2, ..., a_m]}      // one area per member, each in [a_min, a_max]
```
Exactly `m` finite numbers, each within `[a_min, a_max]` (a tiny numerical tolerance is
allowed). Any other shape/type, a non-finite value, or an out-of-range area is rejected.

## FEM model (what the grader computes)
Standard 2D truss: each member contributes the axial element stiffness `(E·A/L)` projected
onto its direction cosines into the global stiffness matrix `K`; the free DOFs are solved
from `K_ff u_f = F_f`; axial stress in a member is `σ = E · (Δu·direction)/L`. The grader
runs this itself on your areas — you never see its internals; you only get the public
instance above.

## Objective & scoring
Minimize total mass `W = Σ rho · L_i · A_i` over feasible designs. Per instance:
`score = min(1, 0.1 · W_baseline / W_yours)`, where `W_baseline` is the mass of the best
**uniform** (single-area-for-all-members) feasible design. Non-uniform designs that follow
the load path can beat the uniform baseline; the final score is the mean over 10 fixed,
seeded instances of varying tower height, thrust, weight, and sway-limit tightness.

## Suggested strategies (increasing sophistication)
- **Uniform sizing** — one area for all members, just big enough to pass both limits.
- **One-shot stress sizing** — size each member to its own axial force, then stiffen
  uniformly to meet the sway limit.
- **Sensitivity / virtual-work sizing** — for a determinate tower the top sway is
  `Σ (N_i n_i L_i)/(E A_i)`; the mass-optimal sizing for that single limit is
  `A_i ∝ sqrt(|N_i n_i|)`, so spend stiffening steel only where it cuts sway per kilogram
  most, and take the per-member max with the fully-stressed area.
