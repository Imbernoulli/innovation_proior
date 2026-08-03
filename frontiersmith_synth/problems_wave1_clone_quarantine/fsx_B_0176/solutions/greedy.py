# TIER: greedy
# Pure node-wise newsvendor, IGNORING the network availability floor. Each station
# is stocked to its own cost-minimizing safety factor k*_i = Phi^{-1}(1 - h_i/p_i),
# clamped to its locker [0, cap_i]. This respects capacities and is cheap, so on
# LOOSE instances (low beta) it is feasible and clearly beats the full-locker
# baseline. But it never looks at the aggregate fill rate, so on TIGHT instances the
# resulting network availability falls below beta and the design is INFEASIBLE -> 0.
import sys, json, math
from statistics import NormalDist

inst = json.load(sys.stdin)
N = inst["N"]; h = inst["h"]; p = inst["p"]; sd = inst["sd"]; cap = inst["cap"]
nd = NormalDist()

stock = []
for i in range(N):
    q = 1.0 - h[i] / p[i]
    k = nd.inv_cdf(q) if 0.0 < q < 1.0 else (0.0 if q <= 0.0 else 8.0)
    s = max(0.0, k) * sd[i]
    stock.append(min(s, cap[i]))
print(json.dumps({"stock": stock}))
