# Trabecular Growth: Load-Adaptive Bone Remodeling Without the Checkerboard

A trabecular bone cross-section is modeled as an `nx` by `ny` grid of cells
anchored at the joint (the entire left column of nodes, `i=0`, is rigidly
fixed) and free elsewhere. Each cell `e` (row-major index `e = cy*nx+cx`,
`cx in [0,nx)`, `cy in [0,ny)`) has a local bone-volume-fraction density
`rho_e` in `[rho_min, 1]`. Cells deform under load as standard bilinear
(Q4) plane-stress finite elements with SIMP stiffness interpolation
`E(rho) = emin_ratio + (1-emin_ratio) * rho^p` (`emin_ratio=1e-3`,
`p` given in the instance). Several distinct seeded **load cases** — different
physiological postures — each apply one point force `(fx, fy)` at a node
`(i=nx, j)` on the free boundary; the true structural compliance is the mean,
over all load cases, of `F^T U` for that case's static solve.

Your program does not solve any FEM itself. It reads the public instance
(`nx, ny, volfrac, p, rho_min, n_iters, k_reg, load_cases`) from stdin and
writes **one JSON answer**:

```json
{"seed_density": [<nx*ny floats in [0,1]>], "filter_radius": <float>, "move_limit": <float>}
```

`seed_density` is the initial field (any total mass; the evaluator projects
it onto the exact target volume `volfrac*nx*ny` via a uniform shift before
remodeling starts). `filter_radius` and `move_limit` define your
**stress-adaptive remodeling law**: the evaluator (frozen) then deterministically
runs `n_iters` steps of a standard optimality-criteria density update. Each
step: solve every load case at the current density, accumulate each cell's
raw strain-energy sensitivity (mean over load cases), optionally smooth that
sensitivity field with a mesh-independent filter of radius `filter_radius`
(cells within the radius are averaged, density-weighted — radius 0 means no
smoothing at all, i.e. purely local stress-following), then move each cell's
density toward the (filtered) sensitivity signal, clipped by `move_limit` per
step and re-projected onto the exact volume budget.

## Scoring — honestly, this is not raw compliance

Purely local stress-following (`filter_radius=0`) chases per-cell sensitivity
noise into spatially incoherent density patterns: cells that look "stiff" in
the coarse per-cell energy sense but are isolated peaks not backed by
comparable-density orthogonal neighbours — a corner-touching checkerboard.
Real trabecular bone cannot form pixel-scale discontinuities; remodeling is a
continuous biological process. The evaluator therefore does not score raw
compliance. It scores

```
obj = compliance(final_density) * (1 + k_reg * roughness(final_density))
```

where `roughness` averages, over every cell with at least one orthogonal
neighbour, `|rho_e - mean(neighbor densities)| / (rho_e + mean + eps)` — 0 for
a perfectly smooth field, larger the more a cell's density disagrees with its
immediate neighbourhood. `k_reg` (given per instance) sets how harshly
incoherent structure is penalized. A remodeling law that spatially regularizes
the sensed signal keeps this near 0 essentially for free; one that does not
pays for it, sometimes badly enough to land worse than never remodeling at all.

Let `B` = the objective of the uniform do-nothing field (`rho_e=volfrac`
everywhere, `roughness=0`) — the evaluator computes this itself. Let
`R = 0.55 * B` (a fixed internal target scale, deliberately better than what
the reference strategies below reach, so there is real headroom above them).
For a feasible answer with true objective `obj`:

```
r = 0.1 + 0.9 * clip((B - obj) / (B - R), 0, 1)
```

so doing nothing scores exactly 0.1, and an answer whose (possibly checkerboarded)
result is even worse than doing nothing is clipped toward 0.

## Feasibility

Any violation — wrong `seed_density` length, non-finite values, densities
outside `[0,1]`, `filter_radius` outside `[0,6]`, `move_limit` outside `[0,1]`,
or a non-positive/non-finite final compliance — makes the answer infeasible
and scores that instance 0.

There is no single dominant strategy: too little filtering wastes material on
incoherent, unsupported structure; too much filtering (or too small a
`move_limit`) never converges to a real load-bearing shape within `n_iters`
steps; instances vary in grid size, volume budget, number of conflicting load
cases, and `k_reg`, so the right amount of spatial regularization must be read
from the instance, not assumed.
