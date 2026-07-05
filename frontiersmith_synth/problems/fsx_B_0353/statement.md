# Aquarium Gantry Pipe-Sizing (feasibility-gated 2D truss FEM)

A public aquarium hangs its heavy water tanks from a steel **cantilever gantry** that bolts to a
wall on the left and reaches out over the exhibit hall. The gantry is a planar (2D) pin-jointed
**truss**: each member is a hollow steel pipe. Downward tank loads hang from the bottom-chord
joints. You are the plumbing/structures engineer: **choose the cross-sectional area of every pipe
member** so the whole gantry is as **light** (cheap) as possible while staying **safe**.

A design is *safe* (feasible) only if, under the given loads, the pin-jointed FEM says:
- every member's axial stress `|N_e| / A_e` is within the allowable `sigma_allow`, AND
- the largest joint displacement magnitude is within `disp_allow`, AND
- every area is within the manufacturable range `[Amin, Amax]`.

An unsafe design scores **0** (hard feasibility gate). A safe design is scored by its total weight
`W = rho * sum_e A_e * L_e` (lower is better).

This is genuinely open-ended: uniformly fattening every pipe is safe but heavy; the real gain comes
from **redistributing material** — thick pipes where forces are large (near the wall), thin pipes
where they are small — while respecting the joint-deflection limit. There is no closed-form optimum.

## Public instance JSON (stdin)
```
{
  "nodes":      [[x, y], ...],          # joint coordinates, meters
  "members":    [[i, j], ...],          # each pipe connects node i and node j  (m members)
  "fixed_dofs": [d, ...],               # global DOFs held at zero (supports). DOF of node k = [2k, 2k+1]
  "loads":      [[d, value], ...],      # external force 'value' (Newtons) applied at global DOF d
  "E":          float,                  # Young's modulus (Pa)
  "rho":        float,                  # density (kg/m^3)
  "sigma_allow":float,                  # allowable |stress| (Pa)
  "disp_allow": float,                  # allowable max joint displacement magnitude (m)
  "Amin":       float,                  # min area (m^2)
  "Amax":       float                   # max area (m^2)
}
```

## Answer JSON (stdout)
```
{"areas": [A_0, A_1, ..., A_{m-1}]}     # one cross-sectional area (m^2) per member, in [Amin, Amax]
```

## FEM (pin-jointed, deterministic)
Standard 2D truss direct stiffness. Member `e=(i,j)` with length `L`, direction cosines
`c=(x_j-x_i)/L`, `s=(y_j-y_i)/L`, stiffness `k = E*A_e/L` contributes the usual 4x4 block on DOFs
`[2i, 2i+1, 2j, 2j+1]`. Assemble the global `K`, delete `fixed_dofs`, solve `K_ff u = F_f`. Member
axial force `N_e = (E*A_e/L) * ((u_jx-u_ix) c + (u_jy-u_iy) s)`; member stress `= N_e / A_e`.

## Scoring
For each instance the evaluator (in an isolated parent process) runs the FEM on your areas.
- If `areas` is malformed, out of `[Amin, Amax]`, produces a singular/ill-conditioned system, or
  violates any stress/displacement limit -> **instance score 0**.
- Otherwise `objective = W` and `score = min(1, 0.1 * baseline / W)`, where `baseline` is the weight
  of the all-`Amax` design. Setting every area to `Amax` is safe and scores exactly `0.1`.

Final score = mean of per-instance scores over a fixed, seeded set of gantries (varying bay count,
load pattern, and deflection limit). Higher is better; leave-nothing-on-the-table designs approach
but do not reach 1.0.
