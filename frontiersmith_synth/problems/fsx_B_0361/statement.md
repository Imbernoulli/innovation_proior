# Orbital Debris Capture Boom — Multi-Load-Case Minimum-Mass Strut Sizing

## Scenario
A debris-removal servicer satellite extends a long 2D pin-jointed **capture boom**
(a Pratt cantilever truss anchored to the servicer bus) to reach out and grapple a
tumbling piece of orbital debris. You are the structural optimizer: choose the
cross-section **area of every strut** so the boom is as **light as possible** while
surviving *every* capture manoeuvre.

The boom must remain feasible under **all** load cases *simultaneously*:

- **LC0 — net-drag:** the capture net snags the debris and drags the boom tip down.
- **LC1 — de-tumble recoil:** reeling the debris in pulls the tip axially back toward
  the bus while the reaction lifts the top chord.

For **each** load case, a linear-elastic 2D truss FEM computes the axial stress in
every strut and the displacement of the two **grapple-head (tip) nodes**. A design is
**feasible** only if, across *all* load cases:

1. every strut's `|axial stress| ≤ sigma`, and
2. every monitored (tip) node's displacement magnitude `sqrt(ux² + uy²) ≤ u_max`
   (the grapple must stay pointed at its target).

Lighter boom = better. An infeasible boom is useless and scores **0**.

## Public instance JSON (stdin)
A single JSON object:

| field         | meaning |
|---------------|---------|
| `nodes`       | list of `[x, y]` joint coordinates (metres) |
| `bars`        | list of `[i, j]` strut endpoint node indices (length `m`) |
| `fixed`       | list of constrained global DOF indices (DOF of node `n` are `2n`, `2n+1`) |
| `load_cases`  | list of load cases; each is a list of `[dof, force_N]` pairs |
| `monitor`     | list of node indices whose displacement magnitude is limited by `u_max` |
| `E`           | Young's modulus (Pa) |
| `rho`         | material density (kg/m³) — mass = Σ rho·length·area |
| `sigma`       | allowable axial stress magnitude (Pa) |
| `a_min`,`a_max` | per-strut cross-section area bounds (m²) |
| `u_max`       | grapple-head displacement budget (m) |

## Answer JSON (stdout)
```json
{"areas": [a_0, a_1, ..., a_{m-1}]}
```
A list of exactly `m` cross-section areas (m²), one per strut in `bars` order.
Every area must satisfy `a_min ≤ a ≤ a_max` and be finite.

## Objective & scoring
- **Minimize** total structural mass `W = Σ rho · length_e · area_e`.
- The evaluator runs the FEM for every load case in the parent process (the answer,
  hidden state, and grading logic are never exposed to your program).
- Feasibility gate: any stress or pointing violation, wrong shape, out-of-range /
  non-finite area, or a singular FEM → score `0`.
- A feasible design scores `min(1, 0.1 · W_baseline / W)`, where `W_baseline` is the
  mass of the smallest **uniform** (single-area) boom that passes all constraints.
  A uniform design therefore scores ≈ `0.1`; smarter per-strut sizing scores higher.
  The final score is the mean over 10 held-out boom instances (different lengths,
  heights, capture forces, and pointing budgets — some stress-governed, some
  displacement-governed).

## Your program
Read one public-instance JSON from **stdin**, write one answer JSON to **stdout**.
Your program runs in an isolated subprocess and only ever sees the public instance.
```python
import sys, json
inst = json.load(sys.stdin)
# ... run your own FEM / sizing loop ...
print(json.dumps({"areas": areas}))
```
