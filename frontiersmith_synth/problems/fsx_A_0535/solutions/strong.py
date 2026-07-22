# TIER: strong
# Marginal-cost internalisation. A driver joining route e imposes an externality
# equal to the flow times the MARGINAL latency,  f_e * l_e'(f_e) = a_e*p_e*f_e^p_e.
# Setting tau_e to that externality turns the selfish equilibrium into the social
# optimum -- but the externality must be evaluated at the SOCIAL-OPTIMUM flow, not
# the untolled flow, and it carries the p_e factor (the whole point).
#
# We solve the social optimum at the representative (mean) demand -- equalising the
# marginal social cost a_e*(p_e+1)*f^p_e + b_e -- and price each route at its
# marginal externality there. One fixed toll cannot be optimal for every demand in
# the distribution, so this stays below the per-scenario ideal (headroom remains
# for a policy that fits the toll to the whole demand spread).
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


# social optimum flow at the mean demand: equalise marginal social cost
ap = [a[e] * (p[e] + 1.0) for e in range(m)]
fso = solve(ap, list(b), p, Dbar)
# marginal externality toll at that flow (carries the p_e factor)
tolls = [a[e] * p[e] * fso[e] ** p[e] for e in range(m)]
print(json.dumps({"tolls": tolls}))
