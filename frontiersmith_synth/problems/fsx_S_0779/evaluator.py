#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_S_0779 -- "Trabecular Growth: Load-Adaptive Bone
Remodeling Without the Checkerboard".

Family: trabecular-stress-remodeling (checkerboard-avoidance + compliance-
minimization + stress-adaptive-remodeling), skinned as bone that grows toward
the load. A cantilevered 2D trabecular domain (nx x ny cells, anchored at the
joint = left column of nodes, x=0) is loaded by several distinct physiological
load cases (different postures) applied at nodes on its free boundary. Each
cell e has a local bone-volume-fraction density rho_e in [rho_min, 1]. Elastic
response uses a standard bilinear (Q4) plane-stress finite element per cell
with SIMP stiffness interpolation E(rho) = Emin_ratio + (1-Emin_ratio)*rho^p.

The candidate PROGRAM does not hand-solve any FEM: it submits (i) a seed
density field and (ii) a stress-adaptive REMODELING LAW -- a filter_radius
(spatial smoothing applied to the sensed strain-energy signal before it drives
a density update) and a move_limit (how fast density reacts per step). The
evaluator (frozen) then deterministically iterates n_iters steps of a standard
optimality-criteria density update using the candidate's law, exactly the way
bone senses local strain energy and reshapes deposition/resorption.

THE NOVELTY (checkerboard-avoidance): the reported score is NOT raw structural
compliance. Purely local stress-following (filter_radius=0) drives OC toward
spatially incoherent, high-frequency density patterns -- cells that are locally
"stiff" in the coarse per-element sensitivity but are isolated peaks touching
their support only at single corners, not backed by comparable-density
orthogonal neighbours. Real trabecular bone cannot form such pixel-scale
discontinuities (remodeling is a continuous biological diffusion process), so
the scored objective is

    obj = compliance(final_density) * (1 + k_reg * roughness(final_density))

where roughness averages, over every cell, |rho_e - mean(orthogonal neighbor
densities)| / (rho_e + mean + eps) -- i.e. how sharply a cell's density
disagrees with its immediate neighbourhood (0 for a perfectly smooth field).
A law that filters the sensitivity signal before updating density (spatial
regularization) naturally keeps this near 0; a purely local law does not.

Scoring (deterministic; no wall-time):
  B = baseline(inst) = objective of the uniform (do-nothing) density field
      rho_e = volfrac for all e (roughness 0 there, so B is pure compliance).
  R = ref_frac * B  (ref_frac=0.55, a fixed evaluator-internal target scale --
      a design about 45% cheaper than doing nothing, NOT necessarily hit by
      every strategy, giving headroom above what the reference solutions reach).
  For a FEASIBLE answer with true objective obj:
      r = 0.1 + 0.9 * clip((B - obj) / (B - R), 0, 1)
  -> obj==B (no improvement, e.g. trivial) maps to exactly 0.1; obj==R maps to
     1.0; obj > B (checkerboard so incoherent it is worse than doing nothing)
     is clipped down toward 0. Infeasible / malformed answer -> 0.

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun

P = 3.0
EMIN_RATIO = 1e-3
RHO_MIN = 0.02
N_ITERS = 15
REF_FRAC = 0.55

# (seed, nx, ny, volfrac, k_reg, n_loads)
SPECS = [
    (768, 8, 5, 0.35, 8.0, 3),
    (749, 7, 4, 0.38, 8.0, 2),
    (788, 8, 6, 0.35, 8.0, 4),
    (796, 7, 6, 0.30, 8.0, 3),
    (761, 8, 5, 0.30, 8.0, 2),
    (802, 7, 6, 0.33, 8.0, 4),
    (783, 8, 6, 0.35, 8.0, 3),
    (797, 7, 6, 0.33, 8.0, 3),
    (764, 8, 5, 0.38, 8.0, 2),
    (762, 8, 5, 0.33, 8.0, 2),
]


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    def uf():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (state >> 11) / float(1 << 53)

    nxt.uf = uf
    return nxt


def _node(i, j, nx):
    return j * (nx + 1) + i


def _gen_loads(r, nx, ny, n_loads, mag_lo=6.0, mag_hi=12.0):
    """L point loads on the free (i==nx) boundary, seeded direction+magnitude."""
    loads = []
    for _ in range(n_loads):
        j = r(0, ny)
        ang = r.uf() * 2 * math.pi
        mag = mag_lo + r.uf() * (mag_hi - mag_lo)
        fx = round(mag * math.cos(ang), 4)
        fy = round(mag * math.sin(ang), 4)
        loads.append({"i": nx, "j": j, "fx": fx, "fy": fy})
    return loads


# ----------------------------- Q4 element (E=1, unit square, nu=0.3) -------
def _build_KE(nu=0.3):
    D = [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1 - nu) / 2.0]]
    scale = 1.0 / (1 - nu * nu)
    gp = [-1 / math.sqrt(3), 1 / math.sqrt(3)]
    KE = [[0.0] * 8 for _ in range(8)]
    for xi in gp:
        for eta in gp:
            dN_dxi = [-(1 - eta) / 4, (1 - eta) / 4, (1 + eta) / 4, -(1 + eta) / 4]
            dN_deta = [-(1 - xi) / 4, -(1 + xi) / 4, (1 + xi) / 4, (1 - xi) / 4]
            detJ = 0.25
            dN_dx = [v / 0.5 for v in dN_dxi]
            dN_dy = [v / 0.5 for v in dN_deta]
            B = [[0.0] * 8 for _ in range(3)]
            for i in range(4):
                B[0][2 * i] = dN_dx[i]
                B[1][2 * i + 1] = dN_dy[i]
                B[2][2 * i] = dN_dy[i]
                B[2][2 * i + 1] = dN_dx[i]
            DB = [[sum(D[r_][k] * B[k][c] for k in range(3)) for c in range(8)] for r_ in range(3)]
            for a in range(8):
                for b in range(8):
                    KE[a][b] += sum(B[k][a] * DB[k][b] for k in range(3)) * detJ
    return [[scale * v for v in row] for row in KE]


_KE0 = _build_KE(0.3)


def _solve_linear(A, b):
    n = len(b)
    M = [row[:] for row in A]
    x = b[:]
    for col in range(n):
        piv = col
        best = abs(M[col][col])
        for r_ in range(col + 1, n):
            if abs(M[r_][col]) > best:
                best = abs(M[r_][col]); piv = r_
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
            x[col], x[piv] = x[piv], x[col]
        d = M[col][col]
        if abs(d) < 1e-13:
            d = 1e-13
        for r_ in range(col + 1, n):
            f = M[r_][col] / d
            if f == 0:
                continue
            Mr = M[r_]; Mc = M[col]
            for c in range(col, n):
                Mr[c] -= f * Mc[c]
            x[r_] -= f * x[col]
    for row in range(n - 1, -1, -1):
        s = x[row]
        Mrow = M[row]
        for c in range(row + 1, n):
            s -= Mrow[c] * x[c]
        d = Mrow[row]
        if abs(d) < 1e-13:
            d = 1e-13
        x[row] = s / d
    return x


class _Q4FE:
    def __init__(self, nx, ny, p, emin_ratio, fixed_i=0):
        self.nx, self.ny, self.p, self.emin_ratio = nx, ny, p, emin_ratio
        self.nnx, self.nny = nx + 1, ny + 1
        self.ndof = 2 * self.nnx * self.nny
        self.edof = []
        for ey in range(ny):
            for ex in range(nx):
                n00 = _node(ex, ey, nx); n10 = _node(ex + 1, ey, nx)
                n11 = _node(ex + 1, ey + 1, nx); n01 = _node(ex, ey + 1, nx)
                dofs = []
                for n in (n00, n10, n11, n01):
                    dofs += [2 * n, 2 * n + 1]
                self.edof.append(dofs)
        self.fixed_dofs = set()
        for j in range(self.nny):
            n = _node(fixed_i, j, nx)
            self.fixed_dofs.add(2 * n); self.fixed_dofs.add(2 * n + 1)
        self.free_dofs = [d for d in range(self.ndof) if d not in self.fixed_dofs]
        self.free_index = {d: i for i, d in enumerate(self.free_dofs)}

    def scale(self, rho):
        return self.emin_ratio + (1 - self.emin_ratio) * (rho ** self.p)

    def dscale(self, rho):
        if rho <= 1e-9:
            rho = 1e-9
        return (1 - self.emin_ratio) * self.p * (rho ** (self.p - 1))

    def assemble(self, density):
        nfree = len(self.free_dofs)
        K = [[0.0] * nfree for _ in range(nfree)]
        fi = self.free_index
        for e in range(self.nx * self.ny):
            C = self.scale(density[e])
            dofs = self.edof[e]
            for a in range(8):
                ia = fi.get(dofs[a])
                if ia is None:
                    continue
                Krow = K[ia]; KErow = _KE0[a]
                for b in range(8):
                    ib = fi.get(dofs[b])
                    if ib is None:
                        continue
                    Krow[ib] += C * KErow[b]
        return K

    def solve_case(self, density, load):
        K = self.assemble(density)
        nfree = len(self.free_dofs)
        F = [0.0] * nfree
        fi = self.free_index
        ni = _node(load["i"], load["j"], self.nx)
        dx, dy = 2 * ni, 2 * ni + 1
        if dx in fi:
            F[fi[dx]] += load["fx"]
        if dy in fi:
            F[fi[dy]] += load["fy"]
        Ufree = _solve_linear(K, F)
        compliance = sum(F[i] * Ufree[i] for i in range(nfree))
        return Ufree, compliance

    def element_energy_raw(self, Ufree):
        fi = self.free_index
        se = [0.0] * (self.nx * self.ny)
        for e in range(self.nx * self.ny):
            dofs = self.edof[e]
            u = [Ufree[fi[d]] if d in fi else 0.0 for d in dofs]
            Ku = [sum(_KE0[a][b] * u[b] for b in range(8)) for a in range(8)]
            se[e] = sum(u[a] * Ku[a] for a in range(8))
        return se


def _roughness(density, nx, ny):
    tot = 0.0; n = 0
    for cy in range(ny):
        for cx in range(nx):
            e = cy * nx + cx
            nbrs = []
            if cx > 0: nbrs.append(density[e - 1])
            if cx < nx - 1: nbrs.append(density[e + 1])
            if cy > 0: nbrs.append(density[e - nx])
            if cy < ny - 1: nbrs.append(density[e + nx])
            if not nbrs:
                continue
            nm = sum(nbrs) / len(nbrs)
            tot += abs(density[e] - nm) / (density[e] + nm + 1e-6)
            n += 1
    return tot / n if n else 0.0


def _sensitivity_filter(se, nx, ny, density, radius):
    if radius <= 1e-9:
        return se[:]
    out = [0.0] * len(se)
    cxs = [(e % nx) + 0.5 for e in range(nx * ny)]
    cys = [(e // nx) + 0.5 for e in range(nx * ny)]
    for e in range(nx * ny):
        num = 0.0; den = 0.0
        for i in range(nx * ny):
            H = radius - math.hypot(cxs[i] - cxs[e], cys[i] - cys[e])
            if H <= 0:
                continue
            num += H * density[i] * se[i]
            den += H * density[i]
        rho_e = max(density[e], 1e-6)
        out[e] = num / (rho_e * den) if den > 1e-12 else se[e]
    return out


def _oc_update(density, dc, volfrac, move, rho_min, nele):
    l1, l2 = 1e-12, 1e12
    new = density[:]
    for _ in range(60):
        lmid = 0.5 * (l1 + l2)
        for e in range(nele):
            be = max(1e-12, -dc[e] / lmid)
            cand = density[e] * math.sqrt(be)
            lo = max(rho_min, density[e] - move)
            hi = min(1.0, density[e] + move)
            new[e] = min(hi, max(lo, cand))
        if sum(new) > volfrac * nele:
            l1 = lmid
        else:
            l2 = lmid
    return new


def _project_volume(density, volfrac, rho_min, nele):
    """Uniform additive-shift bisection projecting any density field onto the
    exact target volume (used so any submitted seed_density is made feasible)."""
    lo, hi = -1.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        tot = sum(min(1.0, max(rho_min, d + mid)) for d in density)
        if tot > volfrac * nele:
            hi = mid
        else:
            lo = mid
    s = 0.5 * (lo + hi)
    return [min(1.0, max(rho_min, d + s)) for d in density]


def _run_remodel(nx, ny, p, emin_ratio, rho_min, loads, volfrac, n_iters,
                  seed_density, filter_radius, move_limit):
    fe = _Q4FE(nx, ny, p, emin_ratio)
    nele = nx * ny
    density = _project_volume(seed_density, volfrac, rho_min, nele)
    L = len(loads)
    for _ in range(n_iters):
        se_total = [0.0] * nele
        for load in loads:
            Ufree, _ = fe.solve_case(density, load)
            se = fe.element_energy_raw(Ufree)
            for e in range(nele):
                se_total[e] += se[e] / L
        dc_raw = [-fe.dscale(density[e]) * se_total[e] for e in range(nele)]
        dc = _sensitivity_filter(dc_raw, nx, ny, density, filter_radius) if filter_radius > 0 else dc_raw
        density = _oc_update(density, dc, volfrac, move_limit, rho_min, nele)
    comp_final = 0.0
    for load in loads:
        _, comp = fe.solve_case(density, load)
        comp_final += comp / L
    rg = _roughness(density, nx, ny)
    return comp_final, rg, density


# ----------------------------- instance family ------------------------------
def make_instances():
    out = []
    for (seed, nx, ny, volfrac, k_reg, n_loads) in SPECS:
        r = _rng(seed)
        loads = _gen_loads(r, nx, ny, n_loads)
        public = {
            "nx": nx, "ny": ny, "volfrac": volfrac, "p": P, "rho_min": RHO_MIN,
            "n_iters": N_ITERS, "k_reg": k_reg, "load_cases": loads,
        }
        out.append({"public": public, "hidden": {}})
    return out


def baseline(inst):
    pub = inst["public"]
    nx, ny, volfrac = pub["nx"], pub["ny"], pub["volfrac"]
    seed_u = [volfrac] * (nx * ny)
    comp, rg, _ = _run_remodel(nx, ny, pub["p"], EMIN_RATIO, pub["rho_min"], pub["load_cases"],
                                volfrac, 0, seed_u, 0.0, 0.0)
    return comp * (1.0 + pub["k_reg"] * rg)


def score(inst, answer):
    pub = inst["public"]
    nx, ny, volfrac = pub["nx"], pub["ny"], pub["volfrac"]
    nele = nx * ny
    if not isinstance(answer, dict):
        return False, None
    sd = answer.get("seed_density")
    fr = answer.get("filter_radius")
    mv = answer.get("move_limit")
    if not isinstance(sd, list) or len(sd) != nele:
        return False, None
    try:
        sd = [float(x) for x in sd]
        fr = float(fr)
        mv = float(mv)
    except (TypeError, ValueError):
        return False, None
    if not all(math.isfinite(x) for x in sd) or not math.isfinite(fr) or not math.isfinite(mv):
        return False, None
    if any(x < -1e-9 or x > 1.0 + 1e-9 for x in sd):
        return False, None
    if fr < -1e-9 or fr > 6.0 + 1e-9:
        return False, None
    if mv < -1e-9 or mv > 1.0 + 1e-9:
        return False, None
    fr = max(0.0, fr); mv = max(0.0, mv)
    comp, rg, _ = _run_remodel(nx, ny, pub["p"], EMIN_RATIO, pub["rho_min"], pub["load_cases"],
                                volfrac, pub["n_iters"], sd, fr, mv)
    if not math.isfinite(comp) or comp <= 0.0:
        return False, None
    obj = comp * (1.0 + pub["k_reg"] * rg)
    if not math.isfinite(obj) or obj <= 0.0:
        return False, None
    return True, float(obj)


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=20)
        if st != "OK":
            vec.append(0.0); continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok, obj = False, None
        if not ok or obj is None:
            vec.append(0.0); continue
        b = baseline(inst)
        R = REF_FRAC * b
        denom = b - R
        frac = (b - obj) / denom if denom > 1e-12 else 0.0
        frac = max(0.0, min(1.0, frac))
        r = 0.1 + 0.9 * frac
        vec.append(r if (r == r and 0.0 <= r <= 1.0) else 0.0)
    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
