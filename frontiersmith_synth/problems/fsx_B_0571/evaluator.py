import sys, json, math, random, isorun

# ==========================================================================
# fsx_B_0571 -- illconditioned-fixedpoint-solver (Format B, isolated candidate)
# Theme: iterative refinement for fixed-point maps x = G(x) = M x + c whose
#        conditioning ranges from benign (real, well-separated modes) to
#        near-critical (a complex, oscillatory dominant mode with spectral
#        radius close to 1).
#
# The candidate NEVER sees M or c. It sees only a short PLAIN-iteration prefix
# x_0..x_p (public). From that prefix it must diagnose the spectral regime,
# then PRESCRIBE an acceleration scheme -- a relaxation factor omega (scalar
# or per-step schedule) and an Aitken-extrapolation period -- that the
# evaluator (which holds M, c) runs for the remaining budget of G-evaluations,
# continuing from x_p. Objective: MINIMIZE the final fixed-point residual
# ||G(x) - x||_2. A garbage / out-of-range scheme scores 0.
#
# Mechanisms composed into the objective:
#   * adaptive-relaxation-omega : the amplification of every mode is
#       (1-omega)+omega*lambda; the winning omega DIFFERS by regime (over-relax
#       for a real mode, under-relax for a complex one).
#   * aitken-acceleration       : Aitken's Delta^2 annihilates a single real
#       geometric mode almost exactly, but is unstable on an oscillatory mode.
#   * spectral-regime-diagnosis : the regime (real vs complex dominant mode)
#       must be inferred from the observed successive step vectors.
# ==========================================================================

EPS = 1e-300


# ---------- linear algebra (dense, tiny n) ----------
def _matvec(M, x):
    return [sum(Mi[j] * x[j] for j in range(len(x))) for Mi in M]


def _G(M, c, x):
    v = _matvec(M, x)
    return [v[i] + c[i] for i in range(len(x))]


def _residual(M, c, x):
    g = _G(M, c, x)
    return math.sqrt(sum((g[i] - x[i]) ** 2 for i in range(len(x))))


def _solve(A, b):
    """Solve A y = b by Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [[A[i][j] for j in range(n)] for i in range(n)]
    v = list(b)
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-18:
            return None
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
            v[col], v[piv] = v[piv], v[col]
        pv = M[col][col]
        for r in range(col + 1, n):
            f = M[r][col] / pv
            if f == 0.0:
                continue
            for cc in range(col, n):
                M[r][cc] -= f * M[col][cc]
            v[r] -= f * v[col]
    y = [0.0] * n
    for i in range(n - 1, -1, -1):
        acc = v[i]
        for j in range(i + 1, n):
            acc -= M[i][j] * y[j]
        y[i] = acc / M[i][i]
    return y


# ---------- instance generation ----------
def _orthogonal(rng, n):
    """A deterministic orthogonal matrix via Gram-Schmidt on a random matrix."""
    cols = [[rng.gauss(0.0, 1.0) for _ in range(n)] for _ in range(n)]
    Q = []
    for k in range(n):
        v = cols[k][:]
        for q in Q:
            d = sum(v[i] * q[i] for i in range(n))
            v = [v[i] - d * q[i] for i in range(n)]
        nrm = math.sqrt(sum(vi * vi for vi in v))
        if nrm < 1e-9:
            v = [1.0 if i == k else 0.0 for i in range(n)]
            nrm = 1.0
        Q.append([vi / nrm for vi in v])
    # rows of Q form an orthonormal basis
    return [[Q[i][j] for i in range(n)] for j in range(n)]


def _block_matrix(rng, n, regime, rho, theta, lam1):
    """Build a block-diagonal B (eigenstructure), then rotate by an orthogonal
    similarity so the acting matrix M = O B O^T is dense (no axis-aligned tell)."""
    B = [[0.0] * n for _ in range(n)]
    start = 0
    if regime == "complex":
        a = rho * math.cos(theta)
        b = rho * math.sin(theta)
        B[0][0] = a
        B[0][1] = -b
        B[1][0] = b
        B[1][1] = a
        start = 2
    else:
        B[0][0] = lam1
        start = 1
    for i in range(start, n):
        # sub-dominant real modes: positive & strictly smaller magnitude, so
        # relaxation stays stable and the DOMINANT mode alone sets the regime.
        s = rng.uniform(0.08, 0.5)
        B[i][i] = s
    O = _orthogonal(rng, n)
    # M = O B O^T
    OB = [[sum(O[i][k] * B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    M = [[sum(OB[i][k] * O[j][k] for k in range(n)) for j in range(n)] for i in range(n)]
    return M


def make_instances():
    n = 8
    p = 10         # number of observed plain-iteration steps (iterates x_0..x_10)
    R = 35         # remaining G-eval budget the evaluator runs the scheme for
    # regime, rho (complex modulus), theta (complex angle), lam1 (real dominant)
    specs = [
        ("real", None, None, 0.88),      # benign-ish real
        ("complex", 0.955, 0.62, None),  # near-critical oscillatory  (TRAP)
        ("real", None, None, 0.93),      # slow real
        ("complex", 0.935, 0.45, None),  # oscillatory                (TRAP)
        ("real", None, None, 0.86),
        ("complex", 0.968, 0.80, None),  # near-critical oscillatory  (TRAP)
        ("real", None, None, 0.90),
        ("complex", 0.945, 0.55, None),  # oscillatory                (TRAP)
        ("real", None, None, 0.95),      # very slow real
        ("real", None, None, 0.84),
    ]
    out = []
    for si, (regime, rho, theta, lam1) in enumerate(specs):
        rng = random.Random(571000 + 17 * si)
        M = _block_matrix(rng, n, regime, rho, theta, lam1)
        c = [rng.uniform(-1.0, 1.0) for _ in range(n)]
        x0 = [rng.uniform(-1.0, 1.0) for _ in range(n)]
        iters = [x0]
        x = x0
        for _ in range(p):
            x = _G(M, c, x)
            iters.append(x)
        pub = {
            "n": n,
            "iterates": [[round(v, 12) for v in xk] for xk in iters],
            "remaining_steps": R,
            "omega_min": 0.0,
            "omega_max": 2.0,
        }
        out.append({"public": pub, "hidden": {"M": M, "c": c}})
    return out


# ---------- scheme execution (evaluator side; holds M, c) ----------
def _run_scheme(M, c, x_start, R, omega_seq, period):
    """Run R relaxed G-steps from x_start. omega_seq(k)->omega for step k.
    Every `period` steps (period>0) apply one componentwise Aitken Delta^2
    extrapolation from the last three iterates. Returns final x (or None)."""
    n = len(x_start)
    x = list(x_start)
    hist = [list(x)]
    for k in range(R):
        w = omega_seq(k)
        g = _G(M, c, x)
        nx = [x[i] + w * (g[i] - x[i]) for i in range(n)]
        if any((v != v) or v in (float("inf"), float("-inf")) for v in nx):
            return None
        x = nx
        hist.append(list(x))
        if period > 0 and (k + 1) % period == 0 and len(hist) >= 3:
            y0, y1, y2 = hist[-3], hist[-2], hist[-1]
            xa = list(y2)
            for i in range(n):
                den = y2[i] - 2.0 * y1[i] + y0[i]
                if abs(den) > 1e-14:
                    xa[i] = y0[i] - (y1[i] - y0[i]) ** 2 / den
            if all((v == v) and v not in (float("inf"), float("-inf")) for v in xa):
                x = xa
                hist = [list(x)]      # restart the Delta^2 window
    return x


def baseline(inst):
    """Plain (unrelaxed, no-Aitken) iteration residual -- the do-nothing ceiling."""
    M, c = inst["hidden"]["M"], inst["hidden"]["c"]
    pub = inst["public"]
    x_start = pub["iterates"][-1]
    xf = _run_scheme(M, c, x_start, pub["remaining_steps"], lambda k: 1.0, 0)
    if xf is None:
        return None
    return _residual(M, c, xf)


def score(inst, ans):
    pub = inst["public"]
    R = pub["remaining_steps"]
    lo, hi = pub["omega_min"], pub["omega_max"]
    if not isinstance(ans, dict):
        return False, 0.0
    ow = ans.get("omega", None)
    period = ans.get("aitken_period", 0)
    if not isinstance(period, int) or isinstance(period, bool):
        return False, 0.0
    if period < 0 or period > 10000:
        return False, 0.0
    # omega may be a scalar or a per-step list; validate finiteness + range
    if isinstance(ow, list):
        if len(ow) == 0 or len(ow) > 100000:
            return False, 0.0
        seq = []
        for v in ow:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return False, 0.0
            v = float(v)
            if v != v or v in (float("inf"), float("-inf")):
                return False, 0.0
            if v < lo - 1e-12 or v > hi + 1e-12:
                return False, 0.0
            seq.append(v)
        omega_seq = lambda k: seq[k] if k < len(seq) else seq[-1]
    elif isinstance(ow, (int, float)) and not isinstance(ow, bool):
        w = float(ow)
        if w != w or w in (float("inf"), float("-inf")):
            return False, 0.0
        if w < lo - 1e-12 or w > hi + 1e-12:
            return False, 0.0
        omega_seq = lambda k: w
    else:
        return False, 0.0
    M, c = inst["hidden"]["M"], inst["hidden"]["c"]
    xf = _run_scheme(M, c, pub["iterates"][-1], R, omega_seq, period)
    if xf is None:
        return False, 0.0
    r = _residual(M, c, xf)
    if r != r or r in (float("inf"), float("-inf")):
        return False, 0.0
    return True, r


# scoring-map constants (tuned so trivial==0.10, strong<=0.92, headroom to 0.95)
_S = 0.08           # points per decade of residual reduction over baseline
_CAP = 0.90         # per-instance ceiling (leaves RL headroom above strong)


def _norm(b, obj):
    if b is None:
        return 0.0
    red = math.log10(max(b, EPS) / max(obj, EPS))    # decades better than plain
    v = 0.1 + _S * red
    if v < 0.0:
        v = 0.0
    if v > _CAP:
        v = _CAP
    return v


def main():
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, stt = isorun.run_candidate(cand, inst["public"], timeout=20)
        if stt != "OK":
            vec.append(0.0)
            continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue
        b = baseline(inst)
        r = _norm(b, obj)
        vec.append(r if (r == r and 0 <= r <= 1) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
