# TIER: greedy
# The obvious "congestion charge": run the untolled equilibrium at the mean
# demand, measure how congested each route is, and toll each route by the
# congestion delay it currently shows -- tau_e = a_e * f_e ** p_e.
#
# This is the trap. It looks like internalising congestion, but it drops the
# marginal factor p_e: the real externality a driver imposes is p_e times this.
# On routes with different exponents it under-tolls the steep (high-p) routes
# relative to the flat ones and steers flow the wrong way.
import sys, json

inst = json.load(sys.stdin)
a, b, p = inst["a"], inst["b"], inst["p"]
m = inst["m"]
Dbar = sum(inst["train_demands"]) / len(inst["train_demands"])


def solve(a, beff, p, D, iters=200):
    if D <= 0.0:
        return [0.0] * len(a)
    lo = min(beff)
    hi = min(beff[e] + a[e] * D ** p[e] for e in range(len(a)))
    if hi <= lo:
        hi = lo + 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        s = 0.0
        for e in range(len(a)):
            if mid > beff[e]:
                s += ((mid - beff[e]) / a[e]) ** (1.0 / p[e])
        if s < D:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    return [((lam - beff[e]) / a[e]) ** (1.0 / p[e]) if lam > beff[e] else 0.0
            for e in range(len(a))]


f = solve(a, list(b), p, Dbar)                 # untolled equilibrium flow
tolls = [a[e] * f[e] ** p[e] for e in range(m)]  # congestion level (NO p_e factor)
print(json.dumps({"tolls": tolls}))
