# TIER: strong
# Insight: reserve is not free headroom. Formulate dispatch as a convex
# program: minimize sum_i a_i*(p_i-m_i)^2 + b_i*p_i  s.t. sum_i p_i = D
# (shadow price lambda) and sum_{i in G} p_i <= RHS  (shadow price mu>=0),
# where G = {fast units} U {J} and RHS = capacity of the fast units
# excluding J (the algebraic simplification of the N-1 rule). KKT gives
#   p_i = clip( m_i + (lambda - mu*[i in G] - b_i) / (2*a_i), 0, cap_i ).
# Solve via nested bisection: outer bisection finds the smallest mu>=0 that
# satisfies the reserve cap; inner bisection finds the lambda that matches
# total demand for each trial mu. Because mu is divided by 2*a_i, units
# with small curvature (flat efficiency curves) absorb almost all of the
# required displacement while sharply-peaked units barely move -- exactly
# the curvature-weighted "who pays for reserve" allocation the naive
# proportional repair cannot see.
import sys


def dispatch_for_price(N, caps, ms, as_, bs, price):
    p = [0.0] * N
    for i in range(N):
        pi = ms[i] + (price[i] - bs[i]) / (2.0 * as_[i])
        if pi < 0.0:
            pi = 0.0
        elif pi > caps[i]:
            pi = caps[i]
        p[i] = pi
    return p


def econ_lambda(D, N, caps, ms, as_, bs, mu, Gmask, iters=70):
    def total(lam):
        price = [lam - (mu if Gmask[i] else 0.0) for i in range(N)]
        p = dispatch_for_price(N, caps, ms, as_, bs, price)
        return sum(p), p

    lo, hi = -1e7, 1e7
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        tot, _ = total(mid)
        if tot < D:
            lo = mid
        else:
            hi = mid
    _, p = total((lo + hi) / 2.0)
    return p


def strong_dispatch(D, N, caps, ms, as_, bs, fast, J):
    Gmask = [(fast[i] == 1) or (i == J) for i in range(N)]
    RHS = sum(caps[i] for i in range(N) if fast[i] == 1)

    p0 = econ_lambda(D, N, caps, ms, as_, bs, 0.0, Gmask)
    sumG0 = sum(p0[i] for i in range(N) if Gmask[i])
    if sumG0 <= RHS + 1e-9:
        return p0

    lo_mu, hi_mu = 0.0, 1e7
    for _ in range(70):
        mid = (lo_mu + hi_mu) / 2.0
        p = econ_lambda(D, N, caps, ms, as_, bs, mid, Gmask)
        sumG = sum(p[i] for i in range(N) if Gmask[i])
        if sumG > RHS:
            lo_mu = mid
        else:
            hi_mu = mid
    return econ_lambda(D, N, caps, ms, as_, bs, hi_mu, Gmask)


def main():
    it = iter(sys.stdin.read().split())
    N = int(next(it)); T = int(next(it))
    caps = []; ms = []; as_ = []; bs = []; fast = []
    for _ in range(N):
        caps.append(int(next(it)))
        ms.append(float(next(it)))
        as_.append(float(next(it)))
        bs.append(float(next(it)))
        fast.append(int(next(it)))
    J = int(next(it)) - 1
    D = [float(next(it)) for _ in range(T)]

    out_lines = []
    for t in range(T):
        p = strong_dispatch(D[t], N, caps, ms, as_, bs, fast, J)
        out_lines.append(" ".join("%.6f" % x for x in p))
    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
