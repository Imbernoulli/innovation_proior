# TIER: greedy
# Pure node-wise newsvendor optimum, IGNORING the network fill-rate floor. For each
# depot set the safety factor to the unconstrained cost minimizer
#   k*_i = Phi^{-1}(1 - h_i / p_i)      (critical-ratio / newsvendor rule)
# which balances that depot's own holding vs shortage cost. On "loose" instances
# (high shortage penalty, modest beta) this already meets the network fill rate and
# is cheap -> good score. On "tight" instances the node-wise choice leaves the
# aggregate fill rate below beta, so the whole design is infeasible and scores 0.
import sys, json, math

inst = json.load(sys.stdin)
N = inst["N"]; h = inst["h"]; p = inst["p"]; sd = inst["sd"]

_SQRT2 = math.sqrt(2.0)
def Phi(x): return 0.5 * (1.0 + math.erf(x / _SQRT2))
def invPhi(q):
    if q <= 0.0: return 0.0
    if q >= 1.0: return 8.0
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if Phi(mid) < q: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)

stock = []
for i in range(N):
    ratio = h[i] / p[i]
    k = 0.0 if ratio >= 1.0 else max(0.0, invPhi(1.0 - ratio))
    stock.append(k * sd[i])
print(json.dumps({"stock": stock}))
