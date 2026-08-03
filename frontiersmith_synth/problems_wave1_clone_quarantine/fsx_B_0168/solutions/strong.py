# TIER: strong
# Two-stage constrained design that respects BOTH the per-station life-support (local)
# floors and the programme fill floor at (near) minimum cost.
#   Stage 1 -- local pinning: for each life-support station c, its local floor
#     1 - B_c/mu_c >= alpha  <=>  Loss(k_c) <= (1-alpha)*mu_c/sd_c, so set a hard lower
#     bound lk_c = Loss^{-1}((1-alpha)*mu_c/sd_c) via bisection (0 for non-critical).
#   Stage 2 -- cost-optimal start then network water-filling: begin at
#     k_i = max(lk_i, newsvendor k*_i). If the programme fill is already >= beta this is
#     the cheapest feasible design; otherwise inflate a single uniform penalty
#     (p_i + lambda) -- while never dropping below lk_i -- and bisect lambda upward until
#     the aggregate fill meets beta. Feasible on every instance and close to the
#     Lagrangian optimum.
import sys, json, math

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


def Phi(x):
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def phi(x):
    return math.exp(-0.5 * x * x) / _SQRT2PI


def loss(k):
    if k < 0.0:
        k = 0.0
    return phi(k) - k * (1.0 - Phi(k))


def Phi_inv(u):
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 8.0
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if Phi(mid) < u:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def loss_inv(target):
    # smallest k>=0 with Loss(k) <= target. Loss decreasing from Loss(0)=0.3989.
    if target >= loss(0.0):
        return 0.0
    lo, hi = 0.0, 40.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if loss(mid) > target:
            lo = mid
        else:
            hi = mid
    return hi


def main():
    inst = json.load(sys.stdin)
    N = inst["N"]; sd = inst["sd"]; mu = inst["mu"]; h = inst["h"]; p = inst["p"]
    beta = inst["beta"]; alpha = inst["alpha"]; crit = inst["crit"]
    tot_mu = sum(mu)

    # Stage 1: local floor lower bounds
    lk = [0.0] * N
    for i in range(N):
        if crit[i]:
            target = (1.0 - alpha) * mu[i] / sd[i]
            lk[i] = loss_inv(target)

    # newsvendor cost-optimal per node
    nv = [max(0.0, Phi_inv(1.0 - h[i] / p[i])) for i in range(N)]

    def k_of(lam):
        ks = []
        for i in range(N):
            u = 1.0 - h[i] / (p[i] + lam)
            k = max(0.0, Phi_inv(u))
            if k < lk[i]:
                k = lk[i]
            ks.append(k)
        return ks

    def fill_of(ks):
        B = sum(sd[i] * loss(ks[i]) for i in range(N))
        return 1.0 - B / tot_mu

    # Stage 2 start: max(local floor, newsvendor)
    ks = [max(lk[i], nv[i]) for i in range(N)]
    if fill_of(ks) < beta:
        lo, hi = 0.0, 1.0
        while fill_of(k_of(hi)) < beta and hi < 1e9:
            hi *= 2.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if fill_of(k_of(mid)) < beta:
                lo = mid
            else:
                hi = mid
        ks = k_of(hi)

    stock = [ks[i] * sd[i] for i in range(N)]
    print(json.dumps({"stock": stock}))


main()
