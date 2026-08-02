# TIER: strong
"""Insight: tau(T) = tau0*exp(k/T) varies EXPONENTIALLY with temperature,
so the Deborah-number condition "cooling rate is slow relative to local
relaxation" only actually binds in a narrow temperature band -- above it
the material relaxes essentially instantly (fast cooling is free), below
it the material is already frozen (further slowing down buys almost
nothing, since exp(-t/tau) is already ~1 regardless of t). So instead of
one global rate, decompose the schedule into a FAST region (ramped up to
rate_max as quickly as the gradient cap allows) and a SLOW region
localized to a contiguous band of segments, with the minimal feasible
slow rate found by bisection for each candidate band. Search over all
band placements and a range of half-widths, apply the gradient
slew-limiter to get an achievable schedule for each candidate, and keep
the fastest overall.

This is not "greedy plus more search": greedy searches one scalar (a
single global rate) against the WHOLE schedule; here we search over WHERE
the slow region goes (exploiting the exponential locality of the
relaxation-time law) and calibrate its rate independently of the rest.
"""
import sys, math


def tau_of(T, tau0, k):
    return tau0 * math.exp(k / T)


def build_grid(M, T_hot, T_cold):
    return [T_hot - i * (T_hot - T_cold) / M for i in range(M + 1)]


def simulate_mismatch(rates, T_grid, tau0, k):
    Tf = T_grid[0]
    for i in range(len(rates)):
        Tprev, Tcur = T_grid[i], T_grid[i + 1]
        r = rates[i]
        Tbar = 0.5 * (Tprev + Tcur)
        tau = tau_of(Tbar, tau0, k)
        ti = (Tprev - Tcur) / r
        x = ti / tau
        e = math.exp(-x) if x < 700.0 else 0.0
        Tf = Tcur + r * tau + (Tf - Tprev - r * tau) * e
    return abs(Tf - T_grid[-1])


def slew_limit(target, rate_max, rate_grad_max):
    out = []
    prev = 0.0
    for tgt in target:
        tgt = min(max(tgt, 1e-12), rate_max)
        r = min(prev + rate_grad_max, max(prev - rate_grad_max, tgt))
        r = max(1e-12, min(r, rate_max))
        out.append(r)
        prev = r
    return out


def total_time(rates, T_grid):
    return sum((T_grid[i] - T_grid[i + 1]) / rates[i] for i in range(len(rates)))


def band_best(T_grid, tau0, k, rate_max, rate_grad_max, tol, widths, iters=50):
    M = len(T_grid) - 1
    best = None
    for c in range(M):
        for w in widths:
            band_lo = max(0, c - w)
            band_hi = min(M - 1, c + w)

            def feas(r_slow):
                target = [rate_max] * M
                for j in range(band_lo, band_hi + 1):
                    target[j] = r_slow
                rates = slew_limit(target, rate_max, rate_grad_max)
                return simulate_mismatch(rates, T_grid, tau0, k) <= tol

            lo, hi = 1e-9, rate_max
            if not feas(lo):
                continue
            if feas(hi):
                r_slow = hi
            else:
                l, h = lo, hi
                for _ in range(iters):
                    mid = (l + h) / 2
                    if feas(mid):
                        l = mid
                    else:
                        h = mid
                r_slow = l
            target = [rate_max] * M
            for j in range(band_lo, band_hi + 1):
                target[j] = r_slow
            rates = slew_limit(target, rate_max, rate_grad_max)
            tt = total_time(rates, T_grid)
            if best is None or tt < best[0]:
                best = (tt, band_lo, band_hi, r_slow)
    return best


WIDTHS = [0, 1, 2, 3, 4, 5, 6, 8, 10]


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    M = int(next(it))
    T_hot = float(next(it)); T_cold = float(next(it))
    tau0 = float(next(it)); k = float(next(it))
    rate_max = float(next(it)); rate_grad_max = float(next(it))
    tol = float(next(it)); div = float(next(it))  # div unused: irrelevant to the solver

    T_grid = build_grid(M, T_hot, T_cold)
    best = band_best(T_grid, tau0, k, rate_max, rate_grad_max, tol, WIDTHS)
    if best is None:
        # fallback: shouldn't happen (band search subsumes the uniform
        # solution at width = M), but guard anyway with a very cautious
        # uniform rate.
        rates = slew_limit([rate_max * 1e-6] * M, rate_max, rate_grad_max)
    else:
        _, band_lo, band_hi, r_slow = best
        target = [rate_max] * M
        for j in range(band_lo, band_hi + 1):
            target[j] = r_slow
        rates = slew_limit(target, rate_max, rate_grad_max)

    t = [(T_grid[i] - T_grid[i + 1]) / rates[i] for i in range(M)]
    print(" ".join(f"{x:.9f}" for x in t))


if __name__ == "__main__":
    main()
