import sys, json, math, isorun

# ==========================================================================
# fsx_B_0359 -- multiobjective-hypervolume (Format B, isolated candidate)
# Theme: "quantum lab wiring".  A cryogenic quantum lab must be wired with a
# harness of coax/RF lines routed through the fridge stages.  Each candidate
# WIRING LAYOUT is a decision vector x in [0,1]^n whose M competing costs
# (e.g. signal latency, thermal heat-load, cross-talk) are read off a FROZEN
# deterministic surrogate -- the classic DTLZ2 response surface.  Under a
# FIXED evaluation budget you may submit at most `budget` layouts; the lab
# evaluates all of them and keeps the Pareto-best trade-off set.
#
# OBJECTIVE (max): the exact dominated HYPERVOLUME of the submitted layouts'
# cost vectors, measured against a fixed reference (nadir) point.  A layout is
# only counted if every one of its costs is at or below the reference.  The
# Pareto-optimal front of DTLZ2 is the unit sphere in the positive octant
# (reached only when the "distance" wiring variables sit at 0.5); good solvers
# both push layouts onto that sphere AND spread them to tile the front.
#
# Deterministic: the surrogate + the exact hypervolume are pure functions of
# the candidate's submitted points.  No randomness in scoring.
# ==========================================================================


def dtlz2(x, M, k):
    """DTLZ2 cost vector (minimisation).  x has n = M-1+k entries in [0,1].
    First M-1 are 'angle' vars; last k are 'distance' vars (optimum at 0.5)."""
    n = M - 1 + k
    g = 0.0
    for i in range(M - 1, n):
        d = x[i] - 0.5
        g += d * d
    f = [0.0] * M
    hp = math.pi / 2.0
    # f[0] = (1+g) * prod_{j=0}^{M-2} cos(theta_j)
    prod = 1.0
    for j in range(M - 1):
        prod *= math.cos(x[j] * hp)
    f[0] = (1.0 + g) * prod
    # f[i] = (1+g) * prod_{j=0}^{M-2-i} cos(theta_j) * sin(theta_{M-1-i})
    for i in range(1, M):
        prod = 1.0
        for j in range(0, M - 1 - i):
            prod *= math.cos(x[j] * hp)
        prod *= math.sin(x[M - 1 - i] * hp)
        f[i] = (1.0 + g) * prod
    return f


def hv_rec(points, dims):
    """Exact dominated hypervolume (maximisation, reference at the origin).
    `points`: list of tuples with all coords >= 0.  `dims`: active coord idxs.
    Hypervolume-by-slicing-objectives (HSO); exact for any point set."""
    if not points:
        return 0.0
    if len(dims) == 1:
        d = dims[0]
        return max(p[d] for p in points)
    d = dims[-1]
    rest = dims[:-1]
    order = sorted(points, key=lambda p: p[d], reverse=True)
    total = 0.0
    N = len(order)
    for i in range(N):
        vi = order[i][d]
        vnext = order[i + 1][d] if i + 1 < N else 0.0
        depth = vi - vnext
        if depth <= 0.0:
            continue
        total += depth * hv_rec(order[:i + 1], rest)
    return total


def _hypervolume(cost_vectors, ref, M):
    """Dominated HV of DTLZ2 cost vectors vs reference `ref` (upper bound)."""
    pts = []
    for f in cost_vectors:
        if all(f[i] <= ref[i] + 1e-12 for i in range(M)):
            pts.append(tuple(ref[i] - f[i] for i in range(M)))
    return hv_rec(pts, list(range(M)))


def make_instances():
    # (M objectives, k distance vars, budget, reference value)
    specs = [
        (2, 8, 30, 1.10),
        (2, 10, 20, 1.10),
        (3, 8, 50, 1.10),
        (3, 10, 40, 1.10),
        (3, 10, 25, 1.10),   # harder: tight budget, M=3
        (2, 12, 15, 1.05),   # harder: tight ref + tiny budget
        (3, 12, 50, 1.20),
        (3, 6, 35, 1.10),
        (2, 6, 40, 1.20),
        (3, 10, 50, 1.10),
    ]
    out = []
    for (M, k, budget, Rv) in specs:
        n = M - 1 + k
        pub = {
            "M": M, "k": k, "n": n, "budget": budget,
            "ref": [Rv] * M, "lo": 0.0, "hi": 1.0,
        }
        out.append({"public": pub, "hidden": {}})
    return out


def baseline(inst):
    """Trivial construction the evaluator computes itself: submit the single
    'centre' layout (all wiring vars = 0.5).  It sits on the DTLZ2 sphere but
    covers only one middle trade-off -> a small but positive hypervolume."""
    pub = inst["public"]
    M, k, n = pub["M"], pub["k"], pub["n"]
    x = [0.5] * n
    f = dtlz2(x, M, k)
    return _hypervolume([f], pub["ref"], M)


def score(inst, ans):
    pub = inst["public"]
    M, k, n = pub["M"], pub["k"], pub["n"]
    budget = pub["budget"]
    lo, hi = pub["lo"], pub["hi"]
    if not isinstance(ans, dict) or "points" not in ans:
        return False, 0.0
    pts = ans["points"]
    if not isinstance(pts, list):
        return False, 0.0
    if len(pts) > budget:
        return False, 0.0
    cost_vectors = []
    for p in pts:
        if not isinstance(p, list) or len(p) != n:
            return False, 0.0
        xv = []
        for v in p:
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                return False, 0.0
            v = float(v)
            if v != v or v in (float("inf"), float("-inf")):
                return False, 0.0
            if v < lo - 1e-9 or v > hi + 1e-9:
                return False, 0.0
            xv.append(min(max(v, lo), hi))
        cost_vectors.append(dtlz2(xv, M, k))
    hv = _hypervolume(cost_vectors, pub["ref"], M)
    if hv != hv or hv < 0.0 or hv in (float("inf"), float("-inf")):
        return False, 0.0
    return True, hv


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
        # maximisation analog of the min-form F/B ratio: trivial(obj==b) -> 0.1
        r = min(1.0, 0.1 * obj / max(b, 1e-12))
        vec.append(r if (r == r and 0 <= r <= 1) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
