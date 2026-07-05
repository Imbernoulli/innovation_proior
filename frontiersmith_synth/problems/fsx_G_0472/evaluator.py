#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_G_0472 -- "Saddle Optimizer: Duality-Gap Descent"
(family: ml-optimizer-update; format B, quality-metric; objective = MINIMIZE gap).

THEME.  You must DESIGN THE UPDATE RULE of a first-order optimizer for a
convex-concave SADDLE problem

        min_x  max_y   L(x,y)
            = 1/2 x^T P x  +  x^T B y  -  1/2 y^T Q y  +  a^T x  -  c^T y

with P = P^T > 0  (strongly convex in x) and Q = Q^T > 0 (strongly concave in y).
The unique saddle point (x*, y*) is where both gradients vanish.  Plain gradient
descent-ascent (GDA) is notorious for OSCILLATING or DIVERGING on such coupled
saddles: the bilinear term B rotates the joint gradient field, so a naive optimizer
spirals outward.  Fixing this is the job of the update-rule designer -- optimism,
extrapolation, per-player step sizes, and (negative) momentum are the classic tools.

THE FIXED-FORM UPDATE (the "architecture" the evaluator executes for you).
Starting from x_0, y_0 (given) with gradient memory seeded gx_{-1}=gx_0, gy_{-1}=gy_0,
for t = 0, 1, ..., T-1 the evaluator runs YOUR coefficients through:

    gx_t = P x_t + B y_t + a          # dL/dx  (descend)
    gy_t = B^T x_t - Q y_t - c        # dL/dy  (ascend)
    dx   = (1+theta) gx_t - theta gx_{t-1}     # optimistic extrapolation
    dy   = (1+theta) gy_t - theta gy_{t-1}
    x_{t+1} = x_t - eta_x * dx + alpha * (x_t - x_{t-1})   # + heavy-ball momentum
    y_{t+1} = y_t + eta_y * dy + alpha * (y_t - y_{t-1})

You supply the FOUR CONSTANT COEFFICIENTS (eta_x, eta_y, theta, alpha).  Different
choices realize different classical methods:
    theta=0, alpha=0            -> plain GDA           (often diverges here)
    theta=1, alpha=0            -> optimistic GDA / extra-gradient flavor
    alpha<0                     -> negative momentum   (damps saddle rotation)
    eta_x != eta_y              -> asymmetric per-player steps for ill-conditioning
Because the coefficients are CONSTANT over all T steps, the dynamics are a fixed
linear recurrence: convergence is at best GEOMETRIC (gap ~ rho^T), never exact in
a finite budget -- so even the best possible coefficients leave a positive gap
(headroom), and a poor choice diverges.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "n": N,
             "P": [[...]], "Q": [[...]], "B": [[...]],   # N x N matrices (floats)
             "a": [...], "c": [...],                     # length-N vectors
             "x0": [...], "y0": [...],                   # fixed init, length N
             "T": T}                                     # number of update steps
  stdout: ONE JSON object:
            {"eta_x": float, "eta_y": float, "theta": float, "alpha": float}

  A VALID answer is a dict whose four fields are finite real numbers with
      0 < eta_x, eta_y <= 100,   0 <= theta <= 10,   -1 < alpha < 1.
  Anything else (missing key, wrong type, non-finite, out of range, non-JSON,
  crash, timeout) -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance the evaluator runs YOUR
coefficients through the fixed-form update for T steps and measures the final
DUALITY GAP

    gap(x_T, y_T) = [ max_y L(x_T, y) ] - [ min_x L(x, y_T) ]
                  = L(x_T, y*(x_T)) - L(x*(y_T), y_T)   >= 0,

computed exactly via the closed-form inner optima y*(x)=Q^{-1}(B^T x - c) and
x*(y) = -P^{-1}(B y + a).  Smaller gap = better.  We normalize in log-gap space
against a WEAK reference (plain GDA at a default step -> ~0.1) and an UNREACHABLE
ideal gap:
    Lb = log10(gap of the default GDA baseline)          # weak anchor
    Li = log10(GAP_IDEAL)   (GAP_IDEAL below any reachable gap -> ceiling < 1)
    Lc = log10(gap of the candidate)
    r  = clamp( 0.1 + 0.9 * (Lb - Lc) / max(Lb - Li, 1.0), 0, 1 )
A candidate matching the default baseline scores ~0.1; a candidate that reduces
the gap by orders of magnitude climbs toward (but does not reach) 1.0; a candidate
that diverges scores below 0.1 (clamped at 0).

ISOLATION.  The candidate is untrusted and runs in a FRESH bwrap SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The baseline gap,
GAP_IDEAL, and the closed-form duality-gap computation all happen in THIS parent
process, so a frame-walking / filesystem-snooping candidate learns nothing useful
-- and it can only ever emit four bounded scalars anyway.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun

E0 = 0.05          # default GDA step (the weak reference / trivial config)
GAP_IDEAL = 1e-6   # unreachable ideal gap (kept below any reachable gap -> headroom)
GAP_FLOOR = 1e-12  # numerical floor before taking log10


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = seed & ((1 << 64) - 1)

    def nxt():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (state >> 11) / float(1 << 53)      # in [0,1)

    return nxt


# ----------------------------- linear algebra (pure python) ----------------
def _matvec(M, v):
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def _matTvec(M, v):
    n = len(M)
    return [sum(M[i][j] * v[i] for i in range(n)) for j in range(len(M[0]))]


def _solve(A, b):
    """Solve A x = b by Gaussian elimination with partial pivoting. None if singular."""
    n = len(b)
    M = [row[:] for row in A]
    x = b[:]
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[p][k]) < 1e-15:
            return None
        M[k], M[p] = M[p], M[k]
        x[k], x[p] = x[p], x[k]
        for i in range(k + 1, n):
            f = M[i][k] / M[k][k]
            for j in range(k, n):
                M[i][j] -= f * M[k][j]
            x[i] -= f * x[k]
    for i in range(n - 1, -1, -1):
        s = x[i] - sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / M[i][i]
    return x


# ----------------------------- instance family -----------------------------
def _spd(r, n, mu, cond):
    """Symmetric positive-definite matrix via diagonal dominance, optionally
    ill-conditioned by a geometric coordinate rescaling."""
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            v = (r() * 2 - 1) * 1.0
            A[i][j] = v
            A[j][i] = v
    for i in range(n):
        s = sum(abs(A[i][j]) for j in range(n) if j != i)
        A[i][i] = s + mu + r() * mu
    if cond > 1.0:
        d = [cond ** ((i / (n - 1)) - 0.5) for i in range(n)]
        for i in range(n):
            for j in range(n):
                A[i][j] *= d[i] * d[j]
    return A


def _build_instance(seed, n, mu_x, mu_y, coup, Titers, cond):
    r = _rng(seed)
    P = _spd(r, n, mu_x, cond)
    Q = _spd(r, n, mu_y, cond)
    B = [[(r() * 2 - 1) * coup for _ in range(n)] for _ in range(n)]
    a = [(r() * 2 - 1) for _ in range(n)]
    c = [(r() * 2 - 1) for _ in range(n)]
    return {"name": f"saddle{seed}", "n": n, "P": P, "Q": Q, "B": B,
            "a": a, "c": c, "x0": [0.0] * n, "y0": [0.0] * n, "T": Titers}


def _build_instances():
    # (seed, n, mu_x, mu_y, coupling, T, condition-number-spread)
    specs = [
        (101, 6, 0.10, 0.10, 5.0, 55, 6.0),
        (102, 6, 0.08, 0.12, 6.0, 55, 8.0),
        (103, 8, 0.07, 0.07, 5.0, 60, 10.0),
        (104, 6, 0.12, 0.08, 7.0, 50, 6.0),
        (105, 8, 0.06, 0.06, 6.0, 60, 12.0),
        (106, 6, 0.10, 0.10, 7.0, 55, 8.0),
        (107, 8, 0.07, 0.07, 6.5, 60, 10.0),
        (108, 6, 0.07, 0.10, 8.0, 50, 9.0),
        # harder / larger held-out instances (bigger n, stronger coupling)
        (201, 10, 0.06, 0.06, 5.0, 70, 12.0),
        (202, 8, 0.06, 0.06, 7.5, 65, 12.0),
        (203, 10, 0.07, 0.07, 6.5, 70, 10.0),
        (204, 8, 0.05, 0.07, 9.0, 60, 14.0),
    ]
    return [_build_instance(*s) for s in specs]


# ----------------------------- dynamics + gap ------------------------------
def _L(inst, x, y):
    P, Q, B, a, c = inst["P"], inst["Q"], inst["B"], inst["a"], inst["c"]
    Px = _matvec(P, x)
    Qy = _matvec(Q, y)
    By = _matvec(B, y)
    t = 0.5 * sum(x[i] * Px[i] for i in range(len(x)))
    t += sum(x[i] * By[i] for i in range(len(x)))
    t -= 0.5 * sum(y[i] * Qy[i] for i in range(len(y)))
    t += sum(a[i] * x[i] for i in range(len(x)))
    t -= sum(c[i] * y[i] for i in range(len(y)))
    return t


def _duality_gap(inst, x, y):
    P, Q, B, a, c = inst["P"], inst["Q"], inst["B"], inst["a"], inst["c"]
    BTx = _matTvec(B, x)
    ys = _solve(Q, [BTx[i] - c[i] for i in range(len(c))])
    By = _matvec(B, y)
    xs = _solve(P, [-(By[i] + a[i]) for i in range(len(a))])
    if ys is None or xs is None:
        return None
    g = _L(inst, x, ys) - _L(inst, xs, y)
    if g is None or not math.isfinite(g):
        return None
    return abs(g)


def _simulate(inst, eta_x, eta_y, theta, alpha):
    n = inst["n"]
    P, Q, B, a, c = inst["P"], inst["Q"], inst["B"], inst["a"], inst["c"]
    x = inst["x0"][:]
    y = inst["y0"][:]
    xp = x[:]
    yp = y[:]
    gxp = None
    gyp = None
    for _ in range(inst["T"]):
        Px = _matvec(P, x)
        By = _matvec(B, y)
        BTx = _matTvec(B, x)
        Qy = _matvec(Q, y)
        gx = [Px[i] + By[i] + a[i] for i in range(n)]
        gy = [BTx[i] - Qy[i] - c[i] for i in range(n)]
        if gxp is None:
            gxp = gx[:]
            gyp = gy[:]
        dx = [(1 + theta) * gx[i] - theta * gxp[i] for i in range(n)]
        dy = [(1 + theta) * gy[i] - theta * gyp[i] for i in range(n)]
        xn = [x[i] - eta_x * dx[i] + alpha * (x[i] - xp[i]) for i in range(n)]
        yn = [y[i] + eta_y * dy[i] + alpha * (y[i] - yp[i]) for i in range(n)]
        for v in xn:
            if not math.isfinite(v) or abs(v) > 1e12:
                return None
        for v in yn:
            if not math.isfinite(v) or abs(v) > 1e12:
                return None
        xp, yp = x, y
        gxp, gyp = gx, gy
        x, y = xn, yn
    return x, y


def _gap_of(inst, eta_x, eta_y, theta, alpha):
    """Final duality gap achieved by the coefficients, or a large sentinel."""
    res = _simulate(inst, eta_x, eta_y, theta, alpha)
    if res is None:
        return 1e12
    g = _duality_gap(inst, res[0], res[1])
    if g is None or not math.isfinite(g):
        return 1e12
    return max(g, 0.0)


# ----------------------------- answer validation ---------------------------
def _valid_coeffs(ans):
    if not isinstance(ans, dict):
        return None
    out = {}
    for k in ("eta_x", "eta_y", "theta", "alpha"):
        v = ans.get(k, None)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        v = float(v)
        if not math.isfinite(v):
            return None
        out[k] = v
    if not (0.0 < out["eta_x"] <= 100.0):
        return None
    if not (0.0 < out["eta_y"] <= 100.0):
        return None
    if not (0.0 <= out["theta"] <= 10.0):
        return None
    if not (-1.0 < out["alpha"] < 1.0):
        return None
    return out


# ----------------------------- normalization -------------------------------
def _normalize(base_gap, cand_gap):
    Lb = math.log10(max(base_gap, GAP_FLOOR))
    Li = math.log10(GAP_IDEAL)
    Lc = math.log10(max(cand_gap, GAP_FLOOR))
    denom = Lb - Li
    if denom < 1.0:
        denom = 1.0
    r = 0.1 + 0.9 * (Lb - Lc) / denom
    if not (r == r) or r in (float("inf"), float("-inf")):
        return 0.0
    if r < 0.0:
        return 0.0
    if r > 1.0:
        return 1.0
    return r


# ----------------------------- scoring driver ------------------------------
def _public_view(inst):
    return {"name": inst["name"], "n": inst["n"],
            "P": [row[:] for row in inst["P"]],
            "Q": [row[:] for row in inst["Q"]],
            "B": [row[:] for row in inst["B"]],
            "a": inst["a"][:], "c": inst["c"][:],
            "x0": inst["x0"][:], "y0": inst["y0"][:], "T": inst["T"]}


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        base_gap = _gap_of(inst, E0, E0, 0.0, 0.0)     # weak GDA reference (parent-only)
        public = _public_view(inst)
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            coeffs = _valid_coeffs(ans)
        except Exception:
            coeffs = None
        if coeffs is None:
            vec.append(0.0)
            continue
        cand_gap = _gap_of(inst, coeffs["eta_x"], coeffs["eta_y"],
                           coeffs["theta"], coeffs["alpha"])
        r = _normalize(base_gap, cand_gap)
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
