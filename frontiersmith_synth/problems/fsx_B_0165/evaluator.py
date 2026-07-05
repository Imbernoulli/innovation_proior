import sys, json, math, random, isorun

# ==========================================================================
# fsx_B_0165 -- multiobjective-hypervolume (Format B, isolated candidate)
# Theme: "volcano monitoring". A hazards agency must commission a small
# fleet of monitoring station CONFIGURATIONS. Each configuration is a design
# vector x in [0,1]^n. Its performance is scored on M competing objectives
# via a deterministic DTLZ2 surrogate (all MINIMIZED, e.g. detection
# latency / power draw / blind-spot fraction), scaled by a "site-difficulty"
# term g(x) that is zero only on the true trade-off surface (a unit sphere
# octant). Under a FIXED commissioning budget of B configurations, propose a
# batch of query points that MAXIMIZES the Pareto HYPERVOLUME dominated with
# respect to a published reference point. The evaluator computes the exact
# hypervolume (2D sweep for M=2, 3D slicing for M=3).
#
# Objective: MAXIMIZE hypervolume. A batch that exceeds the budget, is out of
# range, or is malformed scores 0. Trivial (all-identical) batch ~= 0.1.
# ==========================================================================


def _dtlz2(x, M):
    """Deterministic DTLZ2 objective vector (M objectives, minimization).
    x has length n = (M-1) + k; the last k coords are the 'distance' vars;
    the first M-1 are 'position' vars. g=0 (distance vars = 0.5) puts the
    point on the true Pareto surface (unit-radius sphere octant)."""
    n = len(x)
    g = 0.0
    for xi in x[M - 1:]:            # last k distance vars
        g += (xi - 0.5) ** 2
    f = []
    for i in range(M):
        val = 1.0 + g
        for j in range(M - 1 - i):
            val *= math.cos(x[j] * math.pi / 2.0)
        if i > 0:
            val *= math.sin(x[M - 1 - i] * math.pi / 2.0)
        f.append(val)
    return f


def _area_union(proj):
    """Area of the union of axis-aligned rectangles [0,px]x[0,py], px,py>0."""
    if not proj:
        return 0.0
    proj = sorted(proj, reverse=True)      # by px desc, then py desc
    area = 0.0
    cummax = 0.0
    npts = len(proj)
    for i in range(npts):
        x, y = proj[i]
        if y > cummax:
            cummax = y
        nxt = proj[i + 1][0] if i + 1 < npts else 0.0
        area += cummax * (x - nxt)
    return area


def _hv(points, ref):
    """Exact hypervolume dominated by `points` (each an M-vector, minimized)
    with respect to reference `ref` (M-vector). Only points strictly better
    than ref in EVERY objective contribute. M in {2,3}."""
    M = len(ref)
    ps = []
    for f in points:
        p = [ref[d] - f[d] for d in range(M)]
        if all(pi > 0.0 for pi in p):
            ps.append(p)
    if not ps:
        return 0.0
    if M == 2:
        return _area_union([(p[0], p[1]) for p in ps])
    # M == 3: slice along z (3rd objective), accumulate 2D area * dz
    ps.sort(key=lambda p: -p[2])
    vol = 0.0
    for k in range(len(ps)):
        nxt_z = ps[k + 1][2] if k + 1 < len(ps) else 0.0
        dz = ps[k][2] - nxt_z
        if dz <= 0.0:
            continue
        proj = [(ps[j][0], ps[j][1]) for j in range(k + 1)]
        vol += _area_union(proj) * dz
    return vol


def make_instances():
    # (M objectives, k distance vars, B budget, ref0 reference coordinate)
    specs = [
        (2, 5, 20, 1.10),
        (2, 7, 28, 1.20),
        (3, 4, 24, 1.10),
        (3, 5, 30, 1.15),
        (3, 6, 36, 1.10),
        (3, 6, 48, 1.20),
        (3, 7, 25, 1.10),
        (3, 8, 40, 1.25),
        (2, 9, 16, 1.10),
        (3, 5, 32, 1.10),
    ]
    out = []
    for si, (M, k, B, ref0) in enumerate(specs):
        rng = random.Random(4100 + si)        # reserved for deterministic jitter
        n = (M - 1) + k
        ref = [round(ref0, 6)] * M
        pub = {
            "M": M,
            "k": k,
            "n": n,
            "B": B,
            "ref": ref,
            "seed": rng.randint(0, 10 ** 9),  # cosmetic; front is fixed
        }
        out.append({"public": pub, "hidden": {}})
    return out


def baseline(inst):
    """Trivial construction the evaluator computes itself: a SINGLE
    configuration at the domain center (all coords 0.5)."""
    pub = inst["public"]
    M, n, ref = pub["M"], pub["n"], pub["ref"]
    f = _dtlz2([0.5] * n, M)
    hv = _hv([f], ref)
    return hv


def score(inst, ans):
    pub = inst["public"]
    M, n, B, ref = pub["M"], pub["n"], pub["B"], pub["ref"]
    if not isinstance(ans, dict) or "points" not in ans:
        return False, 0.0
    pts = ans["points"]
    if not isinstance(pts, list) or len(pts) == 0 or len(pts) > B:
        return False, 0.0
    fvals = []
    for x in pts:
        if not isinstance(x, list) or len(x) != n:
            return False, 0.0
        clean = []
        for v in x:
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                return False, 0.0
            v = float(v)
            if v != v or v in (float("inf"), float("-inf")):
                return False, 0.0
            if v < -1e-9 or v > 1.0 + 1e-9:
                return False, 0.0
            clean.append(min(1.0, max(0.0, v)))
        fvals.append(_dtlz2(clean, M))
    hv = _hv(fvals, ref)
    if hv != hv or hv < 0.0:
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
        # maximization F/B analog: trivial (obj == baseline) -> 0.1, headroom above
        r = min(1.0, 0.1 * obj / max(b, 1e-12))
        vec.append(r if (r == r and 0 <= r <= 1) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
