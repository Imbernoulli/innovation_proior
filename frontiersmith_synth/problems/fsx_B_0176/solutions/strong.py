# TIER: strong
# Capacity-aware Lagrangian water-filling on the availability constraint. Relax
# "fill >= beta" with a multiplier lambda >= 0: each station becomes a newsvendor
# with an INFLATED shortage penalty (p_i + lambda),
#     k_i(lambda) = Phi^{-1}(1 - h_i / (p_i + lambda)),   stock_i = k_i * sd_i,
# but every stock is then CLAMPED to its hard locker capacity cap_i. Raising lambda
# pushes more service everywhere and monotonically raises the aggregate fill rate,
# so bisect for the SMALLEST lambda meeting beta. The key difference from a plain
# tree: when cheap low-variance caches saturate their tight lockers, they stop
# absorbing extra service, and the bisection automatically re-routes the remaining
# service onto the roomier (more expensive) stations -- the correct KKT response to
# the binding capacities. Feasible on every instance and far cheaper than the
# full-locker baseline.
import sys, json, math
from statistics import NormalDist

inst = json.load(sys.stdin)
N = inst["N"]; h = inst["h"]; p = inst["p"]; sd = inst["sd"]
mu = inst["mu"]; cap = inst["cap"]; beta = inst["beta"]
nd = NormalDist()
_SQRT2PI = math.sqrt(2.0 * math.pi)


def Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def phi(x):
    return math.exp(-0.5 * x * x) / _SQRT2PI


def loss(k):
    if k < 0.0:
        k = 0.0
    return phi(k) - k * (1.0 - Phi(k))


tot_mu = sum(mu)
capk = [cap[i] / sd[i] for i in range(N)]   # locker cap expressed as a safety factor


def ks_of(lam):
    out = []
    for i in range(N):
        ratio = h[i] / (p[i] + lam)
        if ratio >= 1.0:
            k = 0.0
        else:
            k = max(0.0, nd.inv_cdf(1.0 - ratio))
        out.append(min(k, capk[i]))         # clamp to hard locker
    return out


def fill_of(ks):
    return 1.0 - sum(sd[i] * loss(ks[i]) for i in range(N)) / tot_mu


# max achievable fill = all lockers full
full = fill_of(capk)
if full < beta - 1e-9:
    # should not happen (evaluator guarantees feasibility); fall back to full lockers
    print(json.dumps({"stock": [cap[i] for i in range(N)]}))
    sys.exit(0)

if fill_of(ks_of(0.0)) >= beta:
    ks = ks_of(0.0)
else:
    lo, hi = 0.0, 1.0
    while fill_of(ks_of(hi)) < beta and hi < 1e12:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if fill_of(ks_of(mid)) < beta:
            lo = mid
        else:
            hi = mid
    ks = ks_of(hi)

stock = [min(ks[i] * sd[i], cap[i]) for i in range(N)]
print(json.dumps({"stock": stock}))
