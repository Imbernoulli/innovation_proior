# TIER: trivial
# One flat price for every region -- ignores that regions differ in base value
# and population, and ignores the network entirely. Finds the best SINGLE price
# by directly grid-searching the real network equilibrium revenue (not a
# region-independent proxy), so this is a fair "do nothing clever" floor.
import sys, json, math


def sigmoid(z):
    if z > 40.0:
        return 0.0
    if z < -40.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(z))


def fixed_point(base, gamma, spread, W, p, T=250):
    m = len(base)
    x = [0.0] * m
    for _ in range(T):
        xn = [0.0] * m
        for i in range(m):
            v = base[i] + gamma[i] * sum(W[i][j] * x[j] for j in range(m))
            xn[i] = sigmoid((p[i] - v) / spread[i])
        x = xn
    return x


def revenue_at_flat(base, gamma, spread, W, pop, price, m):
    p = [price] * m
    x = fixed_point(base, gamma, spread, W, p)
    return sum(p[i] * x[i] * pop[i] for i in range(m))


inst = json.load(sys.stdin)
base, gamma, spread, pop, m = inst["base"], inst["gamma"], inst["spread"], inst["pop"], inst["m"]
W = inst["W"]

hi = max(base) * 3 + 80
best_p, best_r = 0.0, -1.0
coarse = 200
for k in range(coarse + 1):
    price = hi * k / coarse
    r = revenue_at_flat(base, gamma, spread, W, pop, price, m)
    if r > best_r:
        best_r, best_p = r, price
step = hi / coarse
for _ in range(3):
    lo2, hi2 = max(0.0, best_p - step), best_p + step
    for k in range(21):
        price = lo2 + (hi2 - lo2) * k / 20
        r = revenue_at_flat(base, gamma, spread, W, pop, price, m)
        if r > best_r:
            best_r, best_p = r, price
    step /= 10.0

print(json.dumps({"prices": [best_p] * m}))
