# TIER: strong
# Insight (algebraic-structure-exploit).  The crank pivot O0 is the ONE pivot
# whose link (the crank) makes a full turn, so the crank tip A sweeps a whole
# circle of radius a about O0.  The coupler point P stays a fixed distance
# rho=sqrt(u^2+v^2) from A, and over a full crank turn the offset direction
# sweeps every orientation relative to (A-O0).  Hence the target's distance to
# O0 fills the annulus [ |a-rho|, a+rho ] and, from its robust radial extremes,
#     (rmax+rmin)/2  and  (rmax-rmin)/2   ==  { a , rho }   (as an UNORDERED pair).
# So the crank radius a and the coupler-point reach rho are recovered exactly
# up to a two-way swap -- collapsing the 5-D fit to a small (c, b, offset-angle)
# search seeded at the truth.  (The rocker pivot O1 gives NO such reading: the
# rocker only oscillates, so its radial band is never fully swept.)  The blind
# greedy recipe, which knows none of this, keeps missing the crank scale.
import sys, math
import numpy as np

toks = sys.stdin.read().split()
it = iter(toks)
M = int(next(it))
O0 = np.array([float(next(it)), float(next(it))])
O1 = np.array([float(next(it)), float(next(it))])
T = np.array([[float(next(it)), float(next(it))] for _ in range(M)])
g = float(np.hypot(*(O1 - O0)))

def make_cos_sin(K):
    TH = np.linspace(0.0, 2.0 * np.pi, K, endpoint=False)
    return np.cos(TH), np.sin(TH)


def trace(p, branch, COS, SIN):
    a, b, c, u, v = p
    if a <= 0 or b <= 0 or c <= 0:
        return None
    Ax = O0[0] + a * COS
    Ay = O0[1] + a * SIN
    dx = O1[0] - Ax
    dy = O1[1] - Ay
    D = np.hypot(dx, dy)
    ok = (D <= b + c) & (D >= abs(b - c)) & (D > 0)
    if not ok.any():
        return None
    Dc = np.where(D > 0, D, 1.0)
    t = (b * b - c * c + D * D) / (2.0 * Dc)
    h2 = b * b - t * t
    ok = ok & (h2 >= 0)
    h = np.sqrt(np.clip(h2, 0, None))
    fx = Ax + t * dx / Dc
    fy = Ay + t * dy / Dc
    ppx = -dy / Dc
    ppy = dx / Dc
    Bx = fx + branch * h * ppx
    By = fy + branch * h * ppy
    ex = Bx - Ax
    ey = By - Ay
    L = np.hypot(ex, ey)
    L = np.where(L > 0, L, 1.0)
    ex /= L
    ey /= L
    Px = Ax + u * ex + v * (-ey)
    Py = Ay + u * ey + v * ex
    return np.stack([Px, Py], 1)[ok]


def chamfer(S, Tgt):
    if S is None or len(S) == 0:
        return 1e18
    d = np.sqrt(((Tgt[:, None, :] - S[None, :, :]) ** 2).sum(2))
    return 0.5 * (d.min(1).mean() + d.min(0).mean())


# fast cost for the search (subsampled target + coarse crank resolution);
# accurate cost for the final polish (full target + fine crank resolution).
COS_S, SIN_S = make_cos_sin(150)
COS_F, SIN_F = make_cos_sin(260)
step_sub = max(1, M // 110)
T_SUB = T[::step_sub]


def cost_fast(p):
    return min(chamfer(trace(p, +1.0, COS_S, SIN_S), T_SUB),
               chamfer(trace(p, -1.0, COS_S, SIN_S), T_SUB))


def cost(p):
    return min(chamfer(trace(p, +1.0, COS_F, SIN_F), T),
               chamfer(trace(p, -1.0, COS_F, SIN_F), T))


# --- recover {a, rho} from the crank pivot O0 (robust percentiles) ---
r = np.hypot(T[:, 0] - O0[0], T[:, 1] - O0[1])
rmax, rmin = np.percentile(r, 99.0), np.percentile(r, 1.0)
half_sum = 0.5 * (rmax + rmin)
half_dif = 0.5 * (rmax - rmin)
# the two admissible (a, rho) assignments (swap ambiguity)
assignments = [(half_sum, half_dif), (half_dif, half_sum)]


def descend(p0, steps, step0, rng):
    p = np.array(p0, float)
    cur = cost_fast(p)
    step = step0
    for _ in range(steps):
        cand = p + rng.normal(0.0, 1.0, 5) * step
        if cand[0] > 0 and cand[1] > 0 and cand[2] > 0:
            cc = cost_fast(cand)
            if cc < cur:
                p, cur = cand, cc
                continue
        step *= 0.97
    return p, cur


rng = np.random.RandomState(7)
# coarse search over rocker c, coupler b, offset angle psi (u,v = rho*(cos,sin)),
# for BOTH admissible (a, rho) assignments; a and rho are pinned by the invariant.
cand_seeds = []
for (a0, rho) in assignments:
    if a0 <= 0 or rho <= 0:
        continue
    for c0 in np.linspace(0.4 * g, 1.35 * g, 5):
        for b in np.linspace(0.4 * g, 1.6 * g, 5):
            for psi in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False):
                u = rho * math.cos(psi)
                v = rho * math.sin(psi)
                p = np.array([a0, b, c0, u, v])
                cand_seeds.append((cost_fast(p), p))
cand_seeds.sort(key=lambda t: t[0])

# refine the globally best handful with random descent, then re-rank accurately
best_p, best_c = None, 1e18
for _, sp in cand_seeds[:5]:
    p, cur = descend(sp, 85, 0.05 * g, rng)
    cf = cost(p)
    if cf < best_c:
        best_c, best_p = cf, p.copy()

if best_p is None:  # never triggers on well-formed targets
    best_p = np.array([half_sum, g, 0.8 * g, half_dif, 0.0])

# --- final coordinate polish ---
p = best_p.copy()
cur = cost(p)
step = np.array([0.03 * g] * 5)
for _ in range(55):
    improved = False
    for j in range(5):
        for s in (+1.0, -1.0):
            cand = p.copy()
            cand[j] += s * step[j]
            cc = cost(cand)
            if cc < cur:
                p, cur = cand, cc
                improved = True
    if not improved:
        step *= 0.5
        if step.max() < 1e-4 * g:
            break

print("%.10f %.10f %.10f %.10f %.10f" % tuple(p))
