We want to place $N$ points on the unit sphere $S^2$ so that the closest pair is as far apart as possible — to maximize $\psi(X) = \min_{x \neq y} \operatorname{dist}(x,y)$ over all $N$-point sets and report $d_N = \max_{|X|=N} \psi(X)$, equivalently to pack $N$ congruent spherical caps of largest common radius, or to build the largest $N$-point spherical code. The asymmetry of this problem is the whole difficulty. A *lower* bound is trivial: I draw a good arrangement, measure its closest pair, and I am done — for $N=13$ and $N=14$ the literature already supplies candidate arrangements $P_{13}$ and $P_{14}$ that give about $57.14°$ and $55.67°$. What a picture can never deliver is the other half: a proof that *nothing* beats the candidate, and that it is the only arrangement that ties it. That is an optimization over a $2N$-dimensional manifold of configurations, and what I need is a *finite, machine-checkable certificate* that the maximum of $\psi$ over that continuum is attained exactly at one point. The two classical global bounds both fall short of this in the same way: they bracket, they do not pin. The Fejes Tóth area bound triangulates the sphere into its $2N-4$ Euler faces and replaces each by an equilateral spherical triangle, giving $d_N \le \arccos\!\big(c_N/(1-c_N)\big)$ with $c_N = \cos(\pi N/(3N-6))$; it is sharp only where the sphere literally tiles into equilateral triangles ($N=3,4,6,12$) and is fatally loose elsewhere — at $N=14$ it yields $58.6809°$, and at $N=13$ it gives $60.92°$, which is worse than the free fact $d_{13} < 60°$ from the kissing number. The Delsarte linear-programming bound and its Bachoc–Vallentin semidefinite strengthening are finer and finite — they certify $|C| \le f(1)/f_0$ for any positive-definite combination $f = \sum_k f_k P_k^{(n)}$ of Gegenbauer polynomials that is nonpositive on $[-1, \cos\psi]$, and they solved the kissing number in dimensions $8$ and $24$ — but they bound the wrong quantity: they cap how *many* points fit at a fixed angle, so run in reverse they give only a numeric bracket $d_{13} < 58.5°$, $d_{14} < 56.58°$, strictly above the conjectured values, with no equality and no uniqueness. And the hand enumeration of taut-pair graphs by spherical trigonometry, which settled $N \le 11$, simply does not scale: by $N=13$ there are tens of millions of candidate graphs.

The resolution, which I will call the irreducible-contact-graph method, is to stop hoping a single inequality over the whole sphere will pin the answer, and instead model the optimum *locally* and then discretize. The starting observation, read off the small-$N$ optima long ago, is that a maximal arrangement is *jammed*. Keep only the pairs sitting exactly at the minimum distance $d = \psi(X)$ — the taut pairs — and form the contact graph $CG(X)$ with an edge for each. Slack pairs (distance strictly greater than $d$) impose no constraint on any local motion; only the taut ones do. At a maximum there are precisely two elementary moves that could buy room, and both must be forbidden. A *shift* slides a single point: if any direction increases its distance to all of its taut neighbours, the arrangement was not optimal, so at an optimum no point with neighbours can be shifted. A *Danzer flip* applies to a point $x$ with exactly two taut neighbours $y,z$: such an $x$ lies at one of the two intersections of the two distance-$d$ circles, and reflecting it across the great circle through $y,z$ to the other intersection $x'$ keeps it at distance $d$ from $y$ and $z$; if $x'$ is then strictly farther than $d$ from everyone else, the mirror configuration is no worse and $x'$ is now free to be shifted, so $d$ could rise. An arrangement admitting neither move I call *irreducible*, and for $N > 6$ the optimal taut graph is irreducible. This is the hinge, because irreducibility is a *combinatorial* condition, and combinatorial conditions can be enumerated. Elementary spherical geometry turns it into hard structural constraints: $CG(X)$ is **planar** (two taut arcs of equal length $d$ cannot cross, or one of the four cross-distances would drop below $d$, contradicting minimality); every vertex has degree $0,3,4,$ or $5$ (degree $1$ or $2$ leaves a free shift direction, degree $\ge 6$ needs angular room exceeding $2\pi$); and every face is a **convex equilateral spherical polygon** with at most $\lfloor 2\pi/d_N\rfloor \le 6$ sides — triangle, rhombus, pentagon, or hexagon — since $d > 55°$ in the cases at hand forces $2\pi/d < 7$, with an isolated degree-$0$ point only ever fitting inside a hexagon, at most one per hexagon. The continuum of configurations has collapsed into *finitely many planar graphs*, which is exactly what neither the area bound nor the LP exploited.

From here the method has two stages. The first reduces the millions of graphs to a handful (about $94.7$ million candidates for $N=13$, $\sim1.5$ billion for $N=14$). I generate all of them isomorph-free with `plantri`, screen them on the degree and face-size constraints, and for each survivor write its metric system in the face angles $u_{k,i}$ and the edge length $d$. The angle coordinates are the right choice because once the graph and $d$ are fixed, each face is an equilateral polygon whose shape is determined by its angles, so the angles plus $d$ reconstruct the embedding up to isometry. The system reads: an angle-sum equality $\sum_{i\in I(v)} u_i = 2\pi$ at every vertex; the floor $u_i \ge \alpha(d)$ with the equilateral-triangle angle $\alpha(d) = \arccos\!\big(\cos d/(1+\cos d)\big)$ obtained from the spherical law of cosines $\cos\alpha = \cos d(1-\cos d)/(1-\cos^2 d) = \cos d/(1+\cos d)$; triangular faces forced fully equilateral; rhombi obeying $\cot(u_1/2)\cot(u_2/2) = \cos d$, i.e.
$$u_2 = \rho(u_1,d) = 2\cot^{-1}\!\big(\tan(u_1/2)\cos d\big),$$
derived by drawing a diagonal and eliminating its length between the two isosceles cosine laws, which (since $\rho$ is its own inverse and decreasing) brackets each rhombus angle $\alpha(d) \le u_i \le 2\alpha(d)$; and pentagons and hexagons determined by $d$ and any $m-3$ of their angles, the rest explicit functions of those, plus non-flip inequalities $\zeta_{ij} \ge d$ (and $\lambda \ge d$ for a hexagon holding an isolated point) that encode irreducibility at the face level. These constraints are nonlinear and transcendental, and there are too many graphs to solve them honestly, so I do the cheap one-sided thing: I only ever *delete* a graph, and only when a certified outer relaxation is infeasible. Over the angle window $[d_{\text{lo}}, d_{\text{hi}}]$ — lower end the known construction, upper end the Fejes Tóth or tighter SDP bracket — I enclose $\alpha(d)$, $\rho$, and the polygon angle relations by certified linear envelopes on a cell, turning each cell into a *linear* feasibility problem. If `linprog` proves the outer region empty, the exact system inside is empty too and the graph is dead on that cell; if the LP is feasible, I have learned nothing and must refine. A single coarse pass at level $l=1$ already kills almost everything, dropping $N=13$ from $\sim10^8$ graphs to a few thousand; the survivors are the graphs heavy with pentagons and hexagons, where the envelopes are loosest, and on those I bisect the surviving box and re-linearize on each smaller cell, where the approximations of $\rho$ and the polygon functions are tighter. A graph dies only when every cell of the cover has an empty outer LP. What survives is $\Gamma_N$ together with a few edge-deletion relatives $\Gamma_N^{(i)}$ — combinatorially almost identical to $\Gamma_N$, which is precisely why no window-based relaxation can separate them by feasibility alone.

The second stage separates that handful exactly, showing only $\Gamma_N^{(0)} = CG(P_N)$ attains the maximum. Where the graph's symmetry is clean I chase angles: walking the graph and repeatedly applying the rhombus relation $u_j = \rho(u_i,d)$ and the angle-sums collapses everything to one or two free parameters and a single scalar admissibility function. For $N=13$ this is a function $u_{18}(d)$ that any realization must satisfy $u_{18}(d) \ge \alpha(d)$; it is monotonically decreasing and meets $u_{18}(\delta_{13}) = \alpha(\delta_{13})$ exactly, so the constraint forces $d \le \delta_{13}$ with equality only at $\Gamma^{(0)}$, giving strict separation, and $\delta_{13}$ is analytic — with $a := \alpha(d)$ it solves $2\tan(3\pi/8 - a/2) = (1-2\cos a)/\cos a$ and $\cos d = \cos a/(1-\cos a)$, yielding $a_{13} \approx 69.4051°$ and $\delta_{13} \approx 57.1367°$. The same template reduces two of the $N=14$ relatives to a single parameter $x = (u_2-u_1)/2$ and a monotone scalar ($f_{13}'(0) \approx -2.4587 < 0$) that pins $x=0$, the symmetric $\Gamma_{14}$. Where angle-chasing finds no clean monotone scalar — the remaining two $N=14$ relatives — I fall back on infinitesimal rigidity, which is uniform. If $X$ is maximal it is a constrained maximum of $d$ subject to every taut distance being $\ge d$, so the KKT stationarity conditions supply multipliers $\omega_{ij} \ge 0$, one per taut edge, vanishing on non-edges, satisfying the equilibrium
$$F_i := \sum_{j \neq i} \omega_{ij}\, e_{ij} = 0,$$
where $e_{ij}$ is the unit tangent at $x_i$ pointing along the great circle toward $x_j$ — exactly Connelly's nonnegative equilibrium stress, here arising straight from KKT on the packing optimization. From the refinement step I already hold interval enclosures for every angle, hence for each tangent component $e_{ij} = (c_{ij}, s_{ij})$; the equilibrium $\sum_j \omega_{ij} e_{ij} = 0$ with $\omega \ge 0$ relaxes componentwise to linear inequalities in the $\omega_{ij}$, plus a normalization $\sum \omega_{ij} = 1$ to exclude the trivial zero, and `linprog` checks it. For the two hard $N=14$ relatives the LP is infeasible — no nonnegative normalized equilibrium stress exists within the angle enclosures — so they cannot be maximal, leaving only $\Gamma_{14} = CG(P_{14})$ with $d_{14} = \psi(P_{14}) \approx 55.67057°$, unique up to isometry. Notice how the rejected tools return transformed: the global bounds do not pin the answer but supply the window that makes the per-graph linearization finite, the irreducibility structure discretizes the continuum, the LP becomes a one-sided deletion test, and the same LP, reading KKT as a stress, finishes the proof.

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
