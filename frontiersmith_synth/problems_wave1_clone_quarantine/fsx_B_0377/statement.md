# Festival Stage Roof-Truss Rigging Sizing

You are the structural engineer for a temporary festival main-stage roof. A steel
**parallel-chord Pratt truss** spans the stage between two towers: a **pinned** tower at
the bottom-left and a **roller** tower at the bottom-right. The bottom chord carries the
downward **rigging drops** (line-array speakers, LED-wall panels, moving-head lighting
bars); the windward top corner takes a lateral **gust** load.

The truss is a statically-determinate 2D pin-jointed structure, so the axial *force* in
each member is fixed by the geometry and loads and does **not** depend on the areas you
pick — but the stresses, joint displacements and buckling capacities all do. Your job is
to choose a **cross-sectional area for every member** to **minimize total steel weight**
while satisfying three hard safety gates.

## Objective
Minimize
```
weight(areas) = sum_e  areas[e] * length[e]        (density := 1)
```

## Hard feasibility gates (violate ANY -> your design scores 0)
Let `stress[e]` and the joint displacements come from a linear-elastic direct-stiffness
FEM of the pinned/roller truss under the given `loads` (the evaluator runs this itself).
1. **Yield** — for every member: `|stress[e]| <= sigma`.
2. **Buckling** — for every member in **compression** (`stress[e] < 0`):
   `|stress[e]| <= k_buck * E * areas[e] / length[e]^2`.
   (Euler-type: the allowable compressive stress *grows with area*, so slender / long
   struts must be thickened beyond their fully-stressed area.)
3. **Serviceability (sag)** — every joint's displacement magnitude
   `sqrt(ux^2 + uy^2) <= disp_limit`.
4. **Box** — every area in `[a_min, a_max]`.

A fully-stressed design (`area = |force|/sigma`) is lightest but buckles slender struts;
repairing buckling locally still is not enough, because a light truss can bust the global
sag gate on the tighter instances. There is no single easy optimum.

## Candidate contract (isolated program)
Your program reads ONE JSON "public instance" from **stdin** and writes ONE JSON answer to
**stdout**. It runs in an isolated sandbox and only ever sees the public instance.
```python
import sys, json
inst = json.load(sys.stdin)
# ... compute areas ...
print(json.dumps({"areas": areas}))
```

### Public instance JSON
```
{
  "nodes":   [[x,y], ...],            # node coordinates (2n of them)
  "bars":    [[i,j], ...],            # M members, each a pair of node indices
  "loads":   [[fx,fy], ...],          # applied load vector per node
  "fixed":   [[bx,by], ...],          # per-node DOF fixity (true = fixed)
  "E":       200000.0,                # Young's modulus
  "sigma":   250.0,                   # yield stress
  "k_buck":  0.6,                     # buckling coefficient (see gate 2)
  "disp_limit": <float>,              # serviceability sag limit
  "a_min":   <float>, "a_max": <float>
}
```
The truss is statically determinate, so you can recover member forces by running your own
FEM with all areas = 1 (forces are area-independent); then size each member.

### Answer JSON
```
{"areas": [a_0, a_1, ..., a_{M-1}]}     # one area per member, each in [a_min, a_max]
```

## Scoring (deterministic)
For each instance let `b` = weight of the uniform `a_max` design (heaviest, always
feasible). If your answer is feasible on all gates with weight `obj`:
```
r = min(1, 0.1 * b / obj)
```
so the uniform-`a_max` design scores exactly `0.1`, and a design `k` times lighter than
baseline scores `min(1, 0.1*k)`. An infeasible or malformed answer scores `0`. The final
score is the mean of `r` over a fixed, seeded distribution of 12 stage spans (including
larger and smaller held-out spans). Scoring is fully deterministic.
