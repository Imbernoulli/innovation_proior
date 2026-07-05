# Bakery Mezzanine Rack-Gantry Sizing

A bakery distribution warehouse stores palletized flour and sugar on a steel
**mezzanine rack gantry**. The gantry is a 2D pin-jointed steel truss: a bottom
chord that runs between a pinned column base and a roller base, a top chord (the
loaded deck), verticals, and **two crossing diagonals in every panel** (an
X-brace). Because each panel carries a *redundant* diagonal, the truss is
**statically indeterminate** — the axial force in a member depends not only on the
loads but on the *relative stiffness* (i.e. the chosen cross-sections) of all the
members. Slim a member down and the load it used to carry flows into its stiffer
neighbours.

The deck carries downward pallet loads at every deck joint plus one lateral
forklift/seismic nudge. Steel is expensive, so you must choose the
**cross-sectional area of every member** to make the total steel weight as small as
possible — but the design must survive two hard engineering gates.

You write a **standalone program**: read ONE JSON instance from stdin, write ONE
JSON answer to stdout. Your program is run in an isolated subprocess and only ever
sees the public instance below.

## Input (public instance JSON, from stdin)

A single JSON object:

```
{
  "nodes":  [[x,y], ...],        # joint coordinates (metres)
  "bars":   [[i,j], ...],        # each member connects node i and node j (0-based); M members
  "loads":  [[fx,fy], ...],      # external load vector at each node (kN)
  "fixed":  [[bx,by], ...],      # per node: true => that DOF is a support (fixed), false => free
  "E":      float,               # Young's modulus (same units throughout)
  "sigma":  float,               # allowable axial stress magnitude (yield gate)
  "disp_limit": float,           # allowable joint displacement magnitude (service / sag gate)
  "a_min":  float,               # minimum allowed cross-sectional area for any member
  "a_max":  float                # maximum allowed cross-sectional area for any member
}
```

## Output (answer JSON, to stdout)

```
{ "areas": [a_0, a_1, ..., a_{M-1}] }
```

one finite cross-sectional area per member, in the same order as `bars`, each in the
range `[a_min, a_max]`.

## Feasibility gates (a design is worth 0 unless ALL hold)

With your areas the evaluator runs a linear-elastic FEM (direct stiffness method) and
requires:

1. **Bounds**: `a_min <= a_e <= a_max` for every member `e`.
2. **Yield gate**: every member's axial stress magnitude `<= sigma`.
3. **Sag / service gate**: every joint's displacement magnitude `<= disp_limit`.

Any violation (or a malformed / non-finite answer, or a singular FEM system) makes
the whole design **infeasible** and scores 0 for that instance.

## Objective and scoring

Minimize the total steel weight (density := 1):

```
weight(areas) = sum_e areas[e] * length[e]
```

Let `b` be the weight of the uniform `a_max` design (the heaviest sizing, always
feasible — the evaluator computes it itself). For a **feasible** answer with weight
`obj`, the per-instance score is

```
r = min(1, 0.1 * b / obj)
```

so the trivial uniform-`a_max` design scores exactly `0.1`, and a design `k` times
lighter than that baseline scores `min(1, 0.1*k)`. Infeasible or malformed answers
score `0`. The reported **Ratio** is the mean of `r` over all (public and held-out)
instances.

## Why it is open-ended

Sizing purely by the uniform-frame forces (`area = |force|/sigma`) looks optimal, but
because the frame is indeterminate the forces **redistribute** when you make the areas
non-uniform: the stiffened members attract more load and their stress climbs back over
yield. A good solver must iterate the force redistribution to a self-consistent
fixed point, keep a yield margin, and then trade extra weight against the sag gate,
whose tightness varies from instance to instance. There is no single closed-form
optimum.
