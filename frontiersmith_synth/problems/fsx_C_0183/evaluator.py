# evaluator.py -- Format B (isolated), family: optimization-convergence-analysis
# Theme: deep-sea cable network tensioning as a strongly-monotone convex-concave saddle problem.
#
# A candidate designs a per-iteration STEP-SIZE SCHEDULE for a FIXED first-order
# gradient-descent-ascent (GDA) update run for a fixed iteration budget T from a fixed
# start z0.  The evaluator re-runs the dynamics deterministically and measures the final
# operator (gradient) norm ||G(z_T)|| -- the residual network disequilibrium.  Lower is better.
#
# The candidate is untrusted: it runs in an isolated subprocess (isorun) and sees ONLY the
# public instance.  It cannot output the equilibrium directly -- it can only steer the fixed
# dynamics -- so a finite budget always leaves a positive residual (headroom preserved).
import sys, json, math, random
import isorun


def make_instances():
    # (seed, d, T, mu, scale) -- tuned so trivial=0.1, strong in ~[0.19,0.84] (headroom<1).
    specs = [
        (2001, 5, 20, 1.0, 1.70),
        (2002, 6, 20, 1.0, 2.00),
        (2003, 4, 20, 0.8, 1.30),
        (2004, 7, 22, 1.0, 1.75),
        (2005, 6, 18, 0.9, 1.70),
        (2006, 5, 18, 1.0, 1.85),
        (2007, 8, 20, 1.0, 1.95),
        (2008, 6, 30, 1.0, 2.10),
        (2009, 4, 16, 0.7, 1.20),
        (2010, 7, 24, 1.0, 2.30),
    ]
    out = []
    for seed, d, T, mu, scale in specs:
        rng = random.Random(seed)
        A = [[rng.gauss(0, 1) * scale / math.sqrt(d) for _ in range(d)] for _ in range(d)]
        b = [rng.gauss(0, 1) for _ in range(d)]
        c = [rng.gauss(0, 1) for _ in range(d)]
        z0 = [rng.gauss(0, 1) for _ in range(2 * d)]
        pub = {"d": d, "T": T, "mu": mu, "A": A, "b": b, "c": c, "z0": z0}
        out.append({"public": pub, "hidden": {}})
    return out


def _Gop(pub, z):
    # Monotone operator G(z) = [ grad_x f ; -grad_y f ] for
    #   f(x,y) = x^T A y + b^T x - c^T y + (mu/2)||x||^2 - (mu/2)||y||^2 .
    d = pub["d"]; mu = pub["mu"]; A = pub["A"]; b = pub["b"]; c = pub["c"]
    x = z[:d]; y = z[d:]
    gx = [mu * x[i] + b[i] + sum(A[i][j] * y[j] for j in range(d)) for i in range(d)]
    gy = [mu * y[i] + c[i] - sum(A[j][i] * x[j] for j in range(d)) for i in range(d)]
    return gx + gy


def _norm(v):
    return math.sqrt(sum(t * t for t in v))


def baseline(inst):
    # Trivial construction: take zero steps -> residual is the initial disequilibrium.
    pub = inst["public"]
    return _norm(_Gop(pub, pub["z0"]))


def score(inst, ans):
    pub = inst["public"]; T = pub["T"]; n = 2 * pub["d"]
    if not isinstance(ans, dict) or "steps" not in ans:
        return False, 0.0
    steps = ans["steps"]
    if not isinstance(steps, list) or len(steps) != T:
        return False, 0.0
    fsteps = []
    for e in steps:
        if isinstance(e, bool) or not isinstance(e, (int, float)):
            return False, 0.0
        e = float(e)
        if not math.isfinite(e):
            return False, 0.0
        fsteps.append(e)
    z = list(pub["z0"])
    try:
        for eta in fsteps:
            g = _Gop(pub, z)
            z = [z[i] - eta * g[i] for i in range(n)]
            for zi in z:
                if not math.isfinite(zi):
                    return False, 0.0
        obj = _norm(_Gop(pub, z))
    except (OverflowError, ValueError):
        return False, 0.0
    if not math.isfinite(obj):
        return False, 0.0
    return True, obj


def main():
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
            ok = False
        if not ok:
            vec.append(0.0); continue
        b = baseline(inst)
        r = min(1.0, 0.1 * b / max(obj, 1e-12))
        vec.append(r if (r == r and 0.0 <= r <= 1.0) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
