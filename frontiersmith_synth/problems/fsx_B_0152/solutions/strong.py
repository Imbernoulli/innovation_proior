# TIER: strong
# Constrained min-cost design via Lagrangian water-filling on the network fill-rate
# constraint. Relaxing "fill >= beta" with multiplier lambda >= 0 makes each depot's
# subproblem a newsvendor with an INFLATED shortage penalty (p_i + lambda):
#   k_i(lambda) = Phi^{-1}(1 - h_i / (p_i + lambda))
# Raising lambda uniformly pushes more service (stock) everywhere, monotonically
# increasing the aggregate fill rate. So: if lambda = 0 (the pure newsvendor point)
# already meets beta, use it; otherwise bisect for the SMALLEST lambda that meets
# the network fill rate. That is the exact KKT solution -- feasible on every instance
# and far cheaper than the gold-plated baseline, while pooling extra stock onto the
# cheapest-to-serve depots automatically.
import sys, json, math

inst = json.load(sys.stdin)
N = inst["N"]; h = inst["h"]; p = inst["p"]; sd = inst["sd"]; mu = inst["mu"]
beta = inst["beta"]

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)
def Phi(x): return 0.5 * (1.0 + math.erf(x / _SQRT2))
def phi(x): return math.exp(-0.5 * x * x) / _SQRT2PI
def loss(k):
    if k < 0.0: k = 0.0
    return phi(k) - k * (1.0 - Phi(k))
def invPhi(q):
    if q <= 0.0: return 0.0
    if q >= 1.0: return 8.0
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if Phi(mid) < q: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)

tot_mu = sum(mu)

def ks_of(lam):
    out = []
    for i in range(N):
        ratio = h[i] / (p[i] + lam)
        out.append(0.0 if ratio >= 1.0 else max(0.0, invPhi(1.0 - ratio)))
    return out

def fill_of(ks):
    return 1.0 - sum(sd[i] * loss(ks[i]) for i in range(N)) / tot_mu

if fill_of(ks_of(0.0)) >= beta:
    ks = ks_of(0.0)
else:
    lo, hi = 0.0, 1.0
    while fill_of(ks_of(hi)) < beta and hi < 1e12:
        hi *= 2.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if fill_of(ks_of(mid)) < beta: lo = mid
        else: hi = mid
    ks = ks_of(hi)

print(json.dumps({"stock": [ks[i] * sd[i] for i in range(N)]}))
