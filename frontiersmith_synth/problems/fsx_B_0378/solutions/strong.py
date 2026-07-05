# TIER: strong
# Sample-average approximation (SAA) + budget-projected reallocation local search.
# Draws its OWN scenarios from the announced demand distribution (it never sees the held-out
# realizations), starts from the newsvendor allocation, then greedily shifts buffer between
# stations to cut the SIMULATED penalized objective (holding + transfer + shortage + service floor).
import sys, json, random
from statistics import NormalDist
N = NormalDist(0, 1)

inst = json.load(sys.stdin)
n = inst["n"]; parent = inst["parent"]; mean = inst["mean"]; std = inst["std"]
h = inst["h"]; p = inst["p"]; t = inst["t"]; B = inst["budget"]; tgt = inst["service_target"]

children = [[] for _ in range(n)]
for i, pa in enumerate(parent):
    if pa >= 0:
        children[pa].append(i)

rng = random.Random(12345)
NS = 100
scen = [[max(0.0, rng.gauss(mean[i], std[i])) for i in range(n)] for _ in range(NS)]
LAM = 50000.0

def simulate(q):
    tot_cost = 0.0; tot_dem = 0.0; tot_unmet = 0.0
    for D in scen:
        surplus = [q[i] - D[i] if q[i] > D[i] else 0.0 for i in range(n)]
        deficit = [D[i] - q[i] if D[i] > q[i] else 0.0 for i in range(n)]
        covered = [0.0] * n
        cost = 0.0
        for i in range(n):
            ch = children[i]
            if not ch:
                cost += h[i] * surplus[i]; continue
            td = 0.0
            for c in ch:
                td += deficit[c]
            tr = surplus[i] if surplus[i] < td else td
            if td > 1e-12 and tr > 0.0:
                for c in ch:
                    covered[c] += tr * deficit[c] / td
            cost += h[i] * (surplus[i] - tr) + t[i] * tr
        for i in range(n):
            resid = deficit[i] - covered[i]
            if resid < 0.0:
                resid = 0.0
            cost += p[i] * resid
            tot_unmet += resid; tot_dem += D[i]
        tot_cost += cost
    return tot_cost / NS, 1.0 - tot_unmet / max(tot_dem, 1e-12)

def obj(q):
    c, fill = simulate(q)
    return c + LAM * max(0.0, tgt - fill)

def project(q):
    s = sum(q)
    if s > B:
        f = max(0.0, B / s)
        return [q[i] * f for i in range(n)]
    return list(q)

# start from newsvendor fractile, scaled into budget
z = [N.inv_cdf(p[i] / (p[i] + h[i])) for i in range(n)]
q = [mean[i] + z[i] * std[i] for i in range(n)]
base = sum(mean); tot = sum(q)
if tot > B:
    safe = tot - base; room = B - base
    ff = max(0.0, room / safe) if safe > 0 else 0.0
    q = [mean[i] + ff * (q[i] - mean[i]) for i in range(n)]
q = project([max(0.0, v) for v in q])

best = obj(q)
step = [0.5 * std[i] + 1.0 for i in range(n)]
for _ in range(70):
    grad = []
    for i in range(n):
        d = step[i]; qp = q[:]; qp[i] += d
        grad.append((obj(qp) - best) / d)
    order = sorted(range(n), key=lambda i: grad[i])
    add = order[0]; rem = order[-1]
    d = min(step[add], q[rem])
    if d <= 1e-6:
        for i in range(n):
            step[i] *= 0.6
        if max(step) < 0.05:
            break
        continue
    qn = q[:]; qn[add] += d; qn[rem] -= d
    if qn[rem] < 0.0:
        qn[rem] = 0.0
    qn = project(qn)
    o = obj(qn)
    if o < best - 1e-6:
        q = qn; best = o
    else:
        for i in range(n):
            step[i] *= 0.7
        if max(step) < 0.05:
            break

print(json.dumps({"stock": project([max(0.0, v) for v in q])}))
