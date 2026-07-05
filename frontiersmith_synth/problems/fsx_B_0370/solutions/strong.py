# TIER: strong
# Simulation-based marginal allocation. The candidate only sees the DEMAND
# DISTRIBUTION (per-node mean/std) + tree + costs, not the graded scenarios,
# so it draws its OWN Monte-Carlo demand sample and then allocates the
# stockpile one unit at a time to whichever node gives the largest expected
# cost reduction (holding + escalation + shortage), stopping when no unit
# helps. This naturally discovers per-district safety buffers AND central
# pooling reserves where variance justifies them. Robust but not scenario-
# perfect -> beats proportional sizing while leaving headroom.
import sys, json, math, random


def topo(parent):
    n = len(parent)
    depth = [0] * n
    for i in range(1, n):
        d = 0; j = i
        while parent[j] != -1:
            d += 1; j = parent[j]
        depth[i] = d
    return sorted(range(n), key=lambda i: -depth[i])


def cost(x, scen, parent, ordr, h, e, p, N):
    tot = 0.0
    for d in scen:
        avail = [0] * (N + 1); incoming = [0] * (N + 1); lu = [0] * (N + 1)
        for i in range(1, N + 1):
            xi = x[i]; di = d[i]
            if xi >= di:
                avail[i] = xi - di
            else:
                lu[i] = di - xi
        avail[0] = x[0]
        leftover = 0; esc = 0; ru = 0
        for i in ordr:
            inc = incoming[i]; av = avail[i]
            cov = av if av < inc else inc
            esc += cov
            leftover += av - cov
            pu = (lu[i] if i > 0 else 0) + (inc - cov)
            pp = parent[i]
            if pp != -1:
                incoming[pp] += pu
            else:
                ru = pu
        tot += h * leftover + e * esc + p * ru
    return tot / len(scen)


inst = json.load(sys.stdin)
N = inst["N"]; B = inst["B"]; parent = inst["parent"]
means = inst["means"]; stds = inst["stds"]
h, e, p = inst["h"], inst["e"], inst["p"]
ordr = topo(parent)

rng = random.Random(20250701)
M = 40
scen = []
for _ in range(M):
    d = [0] * (N + 1)
    for i in range(1, N + 1):
        v = means[i] + stds[i] * rng.gauss(0.0, 1.0)
        d[i] = int(round(v)) if v > 0 else 0
    scen.append(d)

x = [0] * (N + 1)
cur = cost(x, scen, parent, ordr, h, e, p, N)
used = 0
while used < B:
    best = -1; best_c = cur
    for i in range(N + 1):
        x[i] += 1
        c = cost(x, scen, parent, ordr, h, e, p, N)
        x[i] -= 1
        if c < best_c - 1e-9:
            best_c = c; best = i
    if best < 0:
        break                    # no unit reduces expected cost -> stop
    x[best] += 1
    used += 1
    cur = best_c

print(json.dumps({"stock": x}))
