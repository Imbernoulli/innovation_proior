# TIER: strong
# Anticipatory insight: work out the SYSTEM-OPTIMAL flow split x* up front (the
# split minimizing total travel time, found by equalizing marginal cost
# MC_e(x) = 3a_e x^2 + 2b_e x + c_e across every edge that should carry flow --
# water-filling via bisection on the shared marginal value mu), then post the
# ONE constant toll per edge that makes x* a fixed point of the crowd's own
# best-response rule (rather than reacting to whichever edge is hot right now).
#
# Fixed-point derivation: this instance's best-response rule sends flow
# proportional to 1/cost, so a state x is a fixed point iff x_e*cost_e is the
# SAME constant K for every edge. Setting K = max_e(x*_e * l_e(x*_e)) and
# toll_e = max(0, K/x*_e - l_e(x*_e)) makes x* exactly that fixed point, so the
# crowd's own selfish reconsideration converges to (and stays at) the
# system optimum instead of chasing a moving hotspot.
import sys, json, math


def l_edge(a, b, c, x):
    return a * x * x + b * x + c


def so_flow(E, N, edges, iters=200):
    def x_of_mu(mu, e):
        a, b, c = edges[e]
        A = 3.0 * a
        B = 2.0 * b
        C = c - mu
        if C >= 0 or A <= 0:
            return 0.0
        disc = B * B - 4 * A * C
        if disc < 0:
            return 0.0
        r = (-B + math.sqrt(disc)) / (2 * A)
        return max(0.0, r)

    lo, hi = 0.0, 1.0
    while sum(x_of_mu(hi, e) for e in range(E)) < N and hi < 1e18:
        hi *= 2.0
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if sum(x_of_mu(mid, e) for e in range(E)) < N:
            lo = mid
        else:
            hi = mid
    mu = (lo + hi) / 2.0
    xs = [x_of_mu(mu, e) for e in range(E)]
    ssum = sum(xs)
    if ssum > 1e-9:
        xs = [v * N / ssum for v in xs]
    return xs


def main():
    inst = json.load(sys.stdin)
    E, T, N = inst["E"], inst["T"], inst["N"]
    edges = [(e["a"], e["b"], e["c"]) for e in inst["edges"]]

    xs = so_flow(E, N, edges)
    prod = [xs[e] * l_edge(edges[e][0], edges[e][1], edges[e][2], xs[e]) for e in range(E)]
    K = max(prod) if prod else 0.0
    toll_e = []
    for e in range(E):
        a, b, c = edges[e]
        if xs[e] > 1e-9:
            toll_e.append(max(0.0, K / xs[e] - l_edge(a, b, c, xs[e])))
        else:
            toll_e.append(0.0)

    tolls = [list(toll_e) for _ in range(T)]
    print(json.dumps({"tolls": tolls}))


main()
