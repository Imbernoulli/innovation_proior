# TIER: strong
# Lagrangian-relaxation policy: bisect a fuel price lambda so that picking, per cut,
# the option maximising (value - lambda*fuel) meets the budget, then greedily repair
# the leftover budget with the most fuel-efficient upgrades. Multiple-choice aware
# (considers intermediate options across the whole trace), so it captures the
# diminishing-returns structure the density greedy misses. Near-optimal, with headroom.
import sys, json
inst = json.load(sys.stdin)
N = inst["n_cuts"]
fuel = inst["fuel"]
value = inst["value"]
B = inst["budget"]


def alloc(lam):
    a = []
    for i in range(N):
        best = 0
        bestval = value[i][0] - lam * fuel[i][0]
        for j in range(1, len(fuel[i])):
            s = value[i][j] - lam * fuel[i][j]
            if s > bestval:
                bestval = s
                best = j
        a.append(best)
    return a


lo, hi = 0.0, 100.0
for _ in range(80):
    mid = (lo + hi) / 2.0
    a = alloc(mid)
    if sum(fuel[i][a[i]] for i in range(N)) > B:
        lo = mid
    else:
        hi = mid

a = alloc(hi)
used = sum(fuel[i][a[i]] for i in range(N))

# greedy repair: spend remaining budget on the best value/fuel upgrade available
improved = True
while improved:
    improved = False
    best = None
    for i in range(N):
        for j in range(len(fuel[i])):
            if j == a[i]:
                continue
            df = fuel[i][j] - fuel[i][a[i]]
            dv = value[i][j] - value[i][a[i]]
            if df > 0 and dv > 0 and used + df <= B:
                d = dv / df
                if best is None or d > best[0]:
                    best = (d, i, j, df)
    if best is not None:
        _, i, j, df = best
        a[i] = j
        used += df
        improved = True

print(json.dumps({"assign": a}))
