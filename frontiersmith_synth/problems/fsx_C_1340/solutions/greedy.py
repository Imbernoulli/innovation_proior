# TIER: greedy
"""The obvious first attempt: 'cooling slowly enough to relax stress at
every temperature is safe' -- so just find the single (uniform) cooling
rate that is as fast as possible while STILL meeting the tolerance
EVERYWHERE, and use it for the whole schedule. This is smarter than the
trivial baseline (it bisects against the real tolerance, not a padded
one) but it is still a single global recipe: it has no notion that the
relaxation-time constraint only actually binds in a narrow temperature
band, so it pays the worst-case (near the band) rate for the ENTIRE
schedule, including the wide stretches of temperature where cooling
could safely have been at rate_max. This is the textbook 'uniformly slow
cooling' trap."""
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


def uniform_feasible_rate(T_grid, tau0, k, rate_max, rate_grad_max, tol, iters=60):
    M = len(T_grid) - 1

    def feas(r):
        rates = slew_limit([r] * M, rate_max, rate_grad_max)
        return simulate_mismatch(rates, T_grid, tau0, k) <= tol

    lo, hi = 1e-9, rate_max
    if not feas(lo):
        return lo
    if feas(hi):
        return hi
    for _ in range(iters):
        mid = (lo + hi) / 2
        if feas(mid):
            lo = mid
        else:
            hi = mid
    return lo


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    M = int(next(it))
    T_hot = float(next(it)); T_cold = float(next(it))
    tau0 = float(next(it)); k = float(next(it))
    rate_max = float(next(it)); rate_grad_max = float(next(it))
    tol = float(next(it)); div = float(next(it))  # div unused: irrelevant to the solver

    T_grid = build_grid(M, T_hot, T_cold)
    r = uniform_feasible_rate(T_grid, tau0, k, rate_max, rate_grad_max, tol)
    rates = slew_limit([r] * M, rate_max, rate_grad_max)
    t = [(T_grid[i] - T_grid[i + 1]) / rates[i] for i in range(M)]
    print(" ".join(f"{x:.9f}" for x in t))


if __name__ == "__main__":
    main()
