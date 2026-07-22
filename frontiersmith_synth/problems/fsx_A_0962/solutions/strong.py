# TIER: strong
# The insight: a region's OWN myopic price ignores what its adoption is worth to
# everyone downstream of it in the influence network. Rank regions by downstream
# revenue-weighted influence, sacrifice the top few below their own myopic price
# (even toward zero) to push their adoption toward 1, then RE-PRICE every other
# region upward against the extra value the cascade now delivers to it. Try a
# small set of (how many regions to sacrifice, how deep) choices and keep the
# one with the best SIMULATED total revenue -- a bounded, structural search, not
# an exhaustive per-coordinate optimizer.
import sys, json, math


def sigmoid(z):
    if z > 40.0:
        return 0.0
    if z < -40.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(z))


def fixed_point(base, gamma, spread, W, p, T=200):
    m = len(base)
    x = [0.0] * m
    for _ in range(T):
        xn = [0.0] * m
        for i in range(m):
            v = base[i] + gamma[i] * sum(W[i][j] * x[j] for j in range(m))
            xn[i] = sigmoid((p[i] - v) / spread[i])
        x = xn
    return x


def revenue(x, p, pop):
    return sum(p[i] * x[i] * pop[i] for i in range(len(p)))


def best_price_for(v, spread, steps=250):
    hi = v * 3 + 50
    best_p, best_r = 0.0, -1.0
    for k in range(steps + 1):
        p = hi * k / steps
        r = p * sigmoid((p - v) / spread)
        if r > best_r:
            best_r, best_p = r, p
    return best_p


def myopic_prices(base, spread):
    return [best_price_for(base[i], spread[i]) for i in range(len(base))]


inst = json.load(sys.stdin)
base, gamma, spread, pop, W = inst["base"], inst["gamma"], inst["spread"], inst["pop"], inst["W"]
m = inst["m"]

# downstream influence: how much region i's adoption is worth to everyone else
infl = [sum(gamma[j] * W[j][i] * pop[j] for j in range(m)) for i in range(m)]
order = sorted(range(m), key=lambda i: -infl[i])

p_my = myopic_prices(base, spread)


def refine(hub_set, frac, rounds=3):
    p = list(p_my)
    for i in hub_set:
        p[i] = p_my[i] * (1.0 - frac)
    for _ in range(rounds):
        x = fixed_point(base, gamma, spread, W, p, T=200)
        v = [base[i] + gamma[i] * sum(W[i][j] * x[j] for j in range(m)) for i in range(m)]
        for i in range(m):
            if i not in hub_set:
                p[i] = best_price_for(v[i], spread[i])
    return p


best_p = p_my
best_R = revenue(fixed_point(base, gamma, spread, W, p_my, T=250), p_my, pop)
for K in (1, 2, 3):
    hub_set = set(order[:K])
    for frac in (0.4, 0.7, 1.0):
        p = refine(hub_set, frac)
        R = revenue(fixed_point(base, gamma, spread, W, p, T=250), p, pop)
        if R > best_R:
            best_R, best_p = R, p

print(json.dumps({"prices": best_p}))
