# TIER: greedy
# Single-Lagrangian water-filling that satisfies the PROGRAMME fill floor (beta) only,
# ignoring the per-station life-support (local) floors. Start from the node-wise
# newsvendor optimum k*_i = Phi^{-1}(1 - h_i/p_i); if the network fill is short of beta,
# inflate a single uniform penalty (p_i + lambda) and bisect lambda upward until the
# aggregate fill meets beta. Cheap and feasible on instances with no / loose critical
# stations, but it starves expensive high-variance life-support posts, so on tight
# critical instances a local floor is violated and the answer is rejected (-> 0 there).
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
    # Acklam-style bisection on the (monotone) standard-normal cdf.
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


def main():
    inst = json.load(sys.stdin)
    N = inst["N"]; sd = inst["sd"]; mu = inst["mu"]; h = inst["h"]; p = inst["p"]
    beta = inst["beta"]
    tot_mu = sum(mu)

    def k_of(lam):
        ks = []
        for i in range(N):
            u = 1.0 - h[i] / (p[i] + lam)
            ks.append(max(0.0, Phi_inv(u)))
        return ks

    def fill_of(ks):
        B = sum(sd[i] * loss(ks[i]) for i in range(N))
        return 1.0 - B / tot_mu

    ks = k_of(0.0)
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
