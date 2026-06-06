# The irreducible-contact-graph method for the Tammes problem

## Problem

Place `N` points on `S^2` maximizing the minimum pairwise distance,
`d_N = max_{|X|=N} psi(X)`, `psi(X) = min_{x != y} dist(x,y)` — equivalently, pack `N` congruent spherical caps of largest radius, or build the largest `N`-point spherical code. The lower bound is a construction; the content is the matching **upper bound and uniqueness**, i.e. an exact, finite, machine-checkable proof of optimality (not a numerically optimized record). The method solves the long-open `N = 13` (`d_13 ≈ 57.1367°`) and `N = 14` (`d_14 ≈ 55.67057°`), each unique up to isometry.

## Key idea

Global analytic bounds only bracket `d_N` — the Fejes Tóth area bound `d_N <= arccos(c_N/(1-c_N))`, `c_N = cos(pi N/(3N-6))` (`N=14 -> 58.6809°`, `N=13 -> 60.92°`, sharp only at `N=3,4,6,12`), and the Delsarte LP / SDP code-cardinality bound (`d_13<58.5°`, `d_14<56.58°`) — and give no uniqueness. The leap is to model the optimum **locally**. Keep only the *taut* pairs (distance exactly `psi(X)`) as a **contact graph** `CG(X)`. An optimal arrangement is **irreducible** (jammed): it admits neither a **shift** of a single point nor a **Danzer flip** (reflecting a 2-contact point across the great circle of its neighbours to gain room). Irreducibility forces, by elementary spherical geometry:

- `CG(X)` is **planar** (equal-length taut arcs cannot cross);
- every vertex has degree `0, 3, 4,` or `5`;
- every face is a **convex equilateral spherical polygon** with `<= floor(2pi/d_N) <= 6` sides (triangle, rhombus, pentagon, hexagon), isolated points only inside hexagons (one per hexagon).

This discretizes the continuum of configurations into **finitely many planar graphs**, attacked one at a time.

## Algorithm

**Lemma 1 — finite list.** Generate *all* planar graphs on `N` vertices satisfying the degree/face constraints, isomorph-free, with `plantri` (`94.7M` for `N=13`, `~1.5B` for `N=14`). For each graph, write its metric system in the face angles `u_{k,i}` and `d`:

- angle-sum `sum_{i in I(v)} u_i = 2pi` at every vertex;
- `u_i >= alpha(d)`, `alpha(d) = arccos(cos d/(1+cos d))` (equilateral-triangle angle);
- triangles equilateral; rhombi `cot(u_1/2)cot(u_2/2) = cos d`, i.e. `u_2 = rho(u_1,d) = 2 cot^{-1}(tan(u_1/2)cos d)`, giving `alpha(d) <= u_i <= 2 alpha(d)`;
- pentagons/hexagons determined by `d` and `m-3` angles, with non-flip inequalities `zeta_{ij} >= d` (and `lambda >= d` for a hexagon holding an isolated point).

These nonlinear constraints are enclosed by **outer linear relaxations** (interval enclosures of `alpha`, `rho`, etc., over certified cells in the angle window `[d_lo, d_hi]` = construction lower end up to Fejes Tóth/SDP upper end) and the resulting **LP feasibility** is tested. If every cell in the cover has an empty outer feasible region, the graph is eliminated. Iterate by **bisecting** surviving cells and re-linearizing; numerical nonlinear solves help identify survivors, but the proof uses the LP certificates plus the final exact separation. Level `l=1` already kills almost all graphs; the survivors are `Gamma_N` and a few edge-deletion relatives `Gamma_N^{(i)}`.

**Lemma 2 — identify and separate.** Show only `Gamma_N^{(0)} = CG(P_N)` attains the max:

- *Symmetry / angle-chasing:* use graph symmetry to reduce to one or two free parameters; build a scalar function (e.g. `u_{18}(d)` for `N=13`, monotone decreasing with `u_{18}(delta_13)=alpha(delta_13)`) whose admissibility forces `d <= delta_N` with equality only at `Gamma^{(0)}`. `delta_13` is analytic: `2 tan(3pi/8 - a/2) = (1-2cos a)/cos a`, `cos d = cos a/(1-cos a)`, `a_13 ≈ 69.4051°`, `delta_13 ≈ 57.1367°`.
- *Rigidity / stress matrix* (uniform fallback, used for the hard `N=14` cases): KKT stationarity for a maximal `X` gives a nonnegative **equilibrium stress** `omega_{ij} >= 0` on edges with `sum_j omega_{ij} e_{ij} = 0` (`e_{ij}` = unit tangent at `x_i` toward `x_j`). Using interval enclosures of `e_{ij}`, this becomes a linear system in `omega`; if `linprog` proves that no nonnegative normalized stress exists, the graph is eliminated.

The survivor is `Gamma_N`, giving the exact `d_N` and uniqueness up to isometry.

## Code

```python
import subprocess
import numpy as np
from numpy.polynomial import polynomial as Pp
from scipy.optimize import linprog

def _pad(poly, size):
    out = np.zeros(size)
    out[:min(size, poly.size)] = poly[:min(size, poly.size)]
    return out

def _row(coeffs, size):
    row = np.zeros(size)
    if isinstance(coeffs, dict):
        for i, value in coeffs.items():
            row[i] = value
        return row
    row[:] = np.asarray(coeffs, dtype=float)
    return row

def _poly_max_on_interval(poly_coeffs, lo, hi):
    points = [lo, hi]
    deriv = Pp.polyder(poly_coeffs)
    if deriv.size:
        for root in Pp.polyroots(deriv):
            if abs(root.imag) < 1e-8 and lo <= root.real <= hi:
                points.append(root.real)
    return np.max(Pp.polyval(np.array(points), poly_coeffs))

def area_upper_bound(num_points):
    cN = np.cos(np.pi * num_points / (3 * num_points - 6))
    return np.arccos(cN / (1.0 - cN))  # 14 -> 58.6809 deg; 13 -> 60.92 deg

def gegenbauer_basis(dimension, max_degree):
    polys = [np.array([1.0])]
    if max_degree >= 1:
        polys.append(np.array([0.0, 1.0]))
    for k in range(1, max_degree):
        tPk = np.concatenate([[0.0], polys[k]])
        nxt = ((2*k + dimension - 2) * tPk - k * _pad(polys[k-1], tPk.size))
        polys.append(nxt / (k + dimension - 2))
    return polys[:max_degree + 1]

def code_cardinality_bound(poly_coeffs, dimension, inner_product_ceiling,
                           offdiag_certifier=None):
    coeffs = np.asarray(poly_coeffs, dtype=float)
    while coeffs.size > 1 and abs(coeffs[-1]) < 1e-14:
        coeffs = coeffs[:-1]
    basis = gegenbauer_basis(dimension, coeffs.size - 1)
    gegenbauer_coeffs = np.zeros(coeffs.size)
    residual = coeffs.copy()
    for k in range(coeffs.size - 1, -1, -1):
        lead = residual[k] / basis[k][k]
        gegenbauer_coeffs[k] = lead
        residual[:k+1] -= lead * _pad(basis[k], k + 1)
    if gegenbauer_coeffs[0] <= 0 or np.any(gegenbauer_coeffs[1:] < -1e-9):
        return None
    if offdiag_certifier is None:
        if _poly_max_on_interval(coeffs, -1.0, inner_product_ceiling) > 1e-9:
            return None
    elif not offdiag_certifier(coeffs, -1.0, inner_product_ceiling):
        return None
    return Pp.polyval(1.0, coeffs) / gegenbauer_coeffs[0]

def triangle_angle(side_length):
    return np.arccos(np.cos(side_length) / (1.0 + np.cos(side_length)))

def opposite_angle(angle, side_length):
    return 2.0 * np.arctan(1.0 / (np.tan(angle / 2.0) * np.cos(side_length)))

def planar_candidate_stream(num_points, generator="plantri", options=("-a",)):
    cmd = [generator, "-q", *options, str(num_points)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for record in proc.stdout:
            if record.strip():
                yield record.rstrip()
        err = proc.stderr.read().decode("utf-8", errors="replace").strip()
        rc = proc.wait()
        if rc:
            raise RuntimeError(f"{' '.join(cmd)} failed with code {rc}: {err}")
    finally:
        if proc.poll() is None:
            proc.kill()

def passes_combinatorial_screen(degrees, face_sizes):
    return (all(deg in (0, 3, 4, 5) for deg in degrees) and
            all(3 <= size <= 6 for size in face_sizes))

def angle_sum_equalities(vertex_incidence, num_variables, angle_offset=0):
    equalities = []
    for incident_angles in vertex_incidence:
        row = np.zeros(num_variables)
        row[[angle_offset + i for i in incident_angles]] = 1.0
        equalities.append((row, 2*np.pi))
    return equalities

def lp_empty(num_variables, bounds, equalities=(), inequalities=()):
    A_eq, b_eq, A_ub, b_ub = [], [], [], []
    for coeffs, rhs in equalities:
        A_eq.append(_row(coeffs, num_variables)); b_eq.append(rhs)
    for coeffs, lo, hi in inequalities:
        row = _row(coeffs, num_variables)
        if hi is not None:
            A_ub.append(row); b_ub.append(hi)
        if lo is not None:
            A_ub.append(-row); b_ub.append(-lo)
    res = linprog(
        np.zeros(num_variables),
        A_ub=np.vstack(A_ub) if A_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=np.vstack(A_eq) if A_eq else None,
        b_eq=np.array(b_eq) if b_eq else None,
        bounds=bounds,
        method="highs",
    )
    return res.status == 2  # only a proven infeasible LP eliminates anything

def eliminate_candidate_by_lp(cells):
    for cell in cells:
        empty = lp_empty(
            cell["num_variables"],
            cell["bounds"],
            cell.get("equalities", ()),
            cell.get("inequalities", ()),
        )
        if not empty:
            return False
    return True

def stress_infeasible(edge_list, direction_boxes, num_points):
    m = len(edge_list)
    inequalities = []
    for i in range(num_points):
        for comp in (0, 1):
            lo_row = np.zeros(m)
            hi_row = np.zeros(m)
            for e, (a, b) in enumerate(edge_list):
                c_lo, c_hi = direction_boxes[e][comp]  # tangent from a to b
                if a == i:
                    lo_row[e], hi_row[e] = c_lo, c_hi
                elif b == i:
                    lo_row[e], hi_row[e] = -c_hi, -c_lo
            inequalities.append((lo_row, None, 0.0))  # sum omega*c_lo <= 0
            inequalities.append((hi_row, 0.0, None))  # 0 <= sum omega*c_hi
    return lp_empty(
        m,
        bounds=[(0.0, None)] * m,
        equalities=[(np.ones(m), 1.0)],
        inequalities=inequalities,
    )
```

`area_upper_bound(14) = 58.6809°`, `area_upper_bound(13) = 60.92°`, `triangle_angle(d_14) = 68.8633°`, `opposite_angle(triangle_angle(d_14), d_14) = 2 triangle_angle(d_14) = 137.7267°` — the metric relations and the search window are exact; the proof is the plantri enumeration driven through `eliminate_candidate_by_lp` and finished by angle-chasing or `stress_infeasible`.
