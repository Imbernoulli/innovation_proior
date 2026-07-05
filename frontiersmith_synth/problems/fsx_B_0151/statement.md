# Mountain-Rescue Relay Gantry — Minimum-Mass Strut Sizing (Format B, isolated)

A mountain-rescue network hangs a relay/antenna payload off the face of a cliff on a
**cantilevered gantry**: a 2D pin-jointed steel truss anchored to the rock at its left
edge and carrying the payload as a downward point load at its far (right) tip. The
gantry's geometry (node positions, which struts connect which nodes, the anchor
supports, and the payload) is fixed and given to you. **Your job is to choose the
cross-sectional area of every strut** so the whole thing is as light as possible while
still standing up under load.

A design is **feasible** only if it passes BOTH hard limits, checked by a linear-elastic
2D truss finite-element solve:

1. **Stress limit** — every strut's axial stress satisfies `|σ_i| ≤ sigma`.
2. **Tip-sway limit** — every nodal displacement magnitude satisfies `|u_d| ≤ u_max`.

An infeasible design (any limit exceeded), or an ill-formed answer, scores **0**. Among
feasible designs, lighter is better. There is no closed-form optimum: sizing purely for
stress can violate the sway limit, and sizing purely for stiffness wastes steel — the
two constraints trade off, and the axial forces redistribute as you change areas.

## Public instance (stdin JSON)
```json
{
  "nodes":  [[x, y], ...],            // node coordinates, meters (index = node id)
  "bars":   [[i, j], ...],            // each strut connects node i to node j
  "fixed":  [dof, ...],               // clamped DOFs (dof = 2*node for x, 2*node+1 for y)
  "loads":  [[dof, value_N], ...],    // external nodal forces, Newtons
  "E":      2.0e11,                    // Young's modulus, Pa
  "rho":    7850.0,                    // density, kg/m^3
  "sigma":  2.5e8,                     // allowable |axial stress|, Pa
  "a_min":  1.0e-4,                    // min cross-sectional area, m^2
  "a_max":  2.0e-2,                    // max cross-sectional area, m^2
  "u_max":  <float>                    // allowable |nodal displacement|, m
}
```
The number of struts is `m = len(bars)`.

## Answer (stdout JSON)
```json
{"areas": [a_1, a_2, ..., a_m]}     // one area per strut, each in [a_min, a_max]
```
Exactly `m` finite numbers, each within `[a_min, a_max]` (a tiny numerical tolerance is
allowed). Any other shape/type, a non-finite value, or an out-of-range area is rejected.

## FEM model (what the grader computes)
Standard 2D truss: each bar contributes the axial element stiffness
`(E·A/L)` projected onto its direction cosines into the global stiffness matrix `K`;
the free DOFs are solved from `K_ff u_f = F_f`; axial stress in a bar is
`σ = E · (Δu·direction)/L`. The grader runs this itself on your areas — you never see
its internals; you only get the public instance above.

## Objective & scoring
Minimize total mass `W = Σ rho · L_i · A_i` over feasible designs. Per instance:
`score = min(1, 0.1 · W_baseline / W_yours)`, where `W_baseline` is the mass of the best
**uniform** (single-area-for-all-struts) feasible design. Non-uniform designs that follow
the load path can beat the uniform baseline; the final score is the mean over 10 fixed,
seeded instances of varying span, height, payload, and sway-limit tightness.

## Suggested strategies (increasing sophistication)
- **Uniform sizing** — one area for all struts, just big enough to pass both limits.
- **One-shot stress sizing** — size each strut to its force at the reference solve, then
  stiffen uniformly for the sway limit.
- **Iterative fully-stressed design** — re-solve and resize repeatedly so material tracks
  the redistributed forces; add minimal stiffness for the sway limit.
- **Sensitivity-based sizing** — spend stiffening material where it reduces tip sway per
  kilogram most, instead of scaling uniformly.
