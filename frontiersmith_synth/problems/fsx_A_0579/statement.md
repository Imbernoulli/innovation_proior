# Seating Chart for Shock Absorbers

## Problem
A slender tower is modelled as `n` stacked floors (a shear building). Floor `j`
(`j = 1..n`) has mass `m_j`; consecutive floors are joined by a linear spring, and
`st_j` is the story stiffness between floor `j` and floor `j-1` (floor 1 is tied to
the ground, the roof is free). Small horizontal motions `x(t)` obey

```
M x'' + C x' + K x = 0
```

where `M = diag(m_j)` and `K` is the tridiagonal stiffness matrix of the springs.
The tower already has light **material damping** `C0 = beta * K` (which naturally
quiets the fast modes). You are handed `K_total` identical **shock-absorber units**
and must bolt them to the floors. Putting `t_j` units on floor `j` adds a grounded
viscous damper of strength `t_j * c_unit` there, so the total damping matrix is

```
C = beta * K + diag(t_j * c_unit).
```

The tower's free response is governed by the eigenvalues `lambda` of the `2n x 2n`
state matrix of the system above (equivalently the roots of
`det(M lambda^2 + C lambda + K) = 0`). Every mode decays like `exp(Re(lambda) t)`.
The building settles as fast as its **slowest-decaying** mode, so the quantity to
maximize is the **decay margin**

```
D(t) = - max_lambda Re(lambda)   (> 0; larger = quieter).
```

## Key effect (read this)
A grounded damper is **not** monotone in strength. For any single mode there is a
critical-damping sweet spot: below it more damping helps, but past it the damped
floor dynamically **detaches** (over-damping) and that mode's eigenvalue creeps
back toward the imaginary axis — the margin *falls*. Because `D` is the *minimum*
decay over the whole mode family, stacking units on the antinodes of one mode
over-damps that mode while leaving the others as the bottleneck. Good seating
charts spread moderate damping across floors where several mode shapes overlap.

## Input (stdin)
```
n K_total T_max
c_unit beta
m_1 st_1
...
m_n st_n
```
`c_unit`, `beta`, `m_j`, `st_j` are positive reals; `n`, `K_total`, `T_max` are
integers with `K_total <= n * T_max`.

## Output (stdout)
`n` integers `t_1 ... t_n` (whitespace-separated): the number of shock-absorber
units on each floor.

## Feasibility
`0 <= t_j <= T_max` for every floor, and `sum_j t_j == K_total` exactly (you must
seat the whole shipment). Any violation, wrong count, non-integer, or a placement
whose decay margin is not a positive finite number scores `0`.

## Objective & Scoring
Maximize the decay margin `D(t)`. The checker assembles the exact state matrix,
computes all eigenvalues deterministically, and reports

```
Ratio = min(1.0, 0.1 * D(t) / D(B))
```

where `B` is the checker's own **antinode-concentration** baseline (all units poured
onto the largest-fundamental-amplitude floors). Reproducing that baseline scores
`0.1`; a placement with ten times its margin caps at `1.0`.

## Constraints
`12 <= n <= 64`, `T_max = 4`, time limit 5 s, memory 512 MB. Scoring is exact and
reproducible (single-threaded eigenvalues).

## Example
With `n = 3`, `K_total = 2`, `T_max = 2`, seating `t = (1, 0, 1)` might yield decay
margin `D = 0.021` while the antinode baseline `B` gives `0.011`, so
`Ratio = 0.1 * 0.021 / 0.011 = 0.191`. (Numbers illustrative only.)
