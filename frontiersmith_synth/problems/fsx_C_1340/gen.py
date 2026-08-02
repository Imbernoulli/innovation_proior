#!/usr/bin/env python3
"""gen.py <testId> -- glass-anneal-schedule instance generator.

Deterministic: everything for a given testId is computed from a fixed,
hardcoded per-case physical-parameter table (PLAN) below -- no external
randomness is used at all, so the instance is bit-for-bit reproducible.

The instance models cooling a glass part from T_hot down to T_cold through
M equal-width temperature segments. The solver must choose a positive
duration t_i for each segment (equivalently a cooling rate r_i =
dT_i / t_i). Two physical mechanisms couple across segments:

  1. stress-relaxation-time: local structural relaxation time
     tau(T) = tau0 * exp(k / T)  (Arrhenius law, evaluated at each
     segment's midpoint temperature and held fixed across the segment).
  2. structural-relaxation-lag: the "fictive" (structural) temperature Tf
     lags the true temperature T; within a segment of constant cooling
     rate r and constant tau, the lag obeys the closed-form recurrence
     implemented in `simulate_mismatch` below (exact solution of
     dTf/dt = (T(t)-Tf)/tau for T(t) affine in t).
  3. cooling-rate-gradient: the oven cannot jump its rate between
     consecutive segments by more than rate_grad_max (rate_0 := 0, the
     oven starts at rest).

We pick, per test case, a temperature band (via `frac`, the fractional
location of the crossing temperature T* in [T_cold,T_hot], and `C`, a
band-steepness multiplier) where tau(T) is comparable to the natural
segment-traversal time -- i.e. where slowing down actually helps. This is
what makes "uniformly slow everywhere" (the obvious approach) waste a lot
of time compared to a schedule that is only slow in that narrow band.

`tol` (the residual mismatch tolerance) and `div` (the strictness divisor
used *only* by the checker's own reference "uniform, extra-cautious"
baseline construction) are CALIBRATED at generation time by directly
searching over the same simulate/bisect machinery a solver would use, so
that a plain uniform-rate schedule lands roughly 3-4x slower in total time
than a schedule that localizes its slow segments to the critical band --
without ever revealing that machinery's *output* (the calibration only
picks numbers; it never leaks a schedule).
"""
import sys, math

# ---------- shared physics/schedule primitives ----------

def tau_of(T, tau0, k):
    return tau0 * math.exp(k / T)


def build_grid(M, T_hot, T_cold):
    return [T_hot - i * (T_hot - T_cold) / M for i in range(M + 1)]


def simulate_mismatch(rates, T_grid, tau0, k):
    """Closed-form per-segment fictive-temperature recurrence. rates[i] is
    the (constant) cooling rate used during segment i. Returns the final
    structural mismatch |Tf_M - T_grid[-1]|."""
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
    """Forward slew-rate-limited tracking of a target rate profile,
    starting from a virtual previous rate of 0 (oven starts at rest)."""
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


def uniform_feasible_rate(T_grid, tau0, k, rate_max, rate_grad_max, tol, iters=60):
    """Max constant target rate r (then slew-limited) whose resulting
    schedule satisfies mismatch <= tol. None if even the slowest rate
    fails."""
    M = len(T_grid) - 1

    def feas(r):
        rates = slew_limit([r] * M, rate_max, rate_grad_max)
        return simulate_mismatch(rates, T_grid, tau0, k) <= tol

    lo, hi = 1e-9, rate_max
    if not feas(lo):
        return None
    if feas(hi):
        return hi
    for _ in range(iters):
        mid = (lo + hi) / 2
        if feas(mid):
            lo = mid
        else:
            hi = mid
    return lo


def band_best(T_grid, tau0, k, rate_max, rate_grad_max, tol, widths, iters=50):
    """Best (min total time) schedule of the form: rate_max everywhere
    except a contiguous band [band_lo,band_hi] held at a bisected minimal
    feasible slow rate, all passed through the gradient slew-limiter.
    Searches all band centers and the given half-widths."""
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


WIDTHS = [0, 1, 2, 3, 4, 5, 6, 8]


def calibrate_tol_for_ratio(T_grid, tau0, k, rate_max, rate_grad_max, target_ratio, n=50):
    """Grid-search tol (log-spaced) so that the uniform-bisected total
    time is ~target_ratio times the band-localized total time."""
    M = len(T_grid) - 1
    fast_rates = slew_limit([rate_max] * M, rate_max, rate_grad_max)
    mm_fast = simulate_mismatch(fast_rates, T_grid, tau0, k)
    lo, hi = mm_fast * 1e-4, mm_fast * 3
    best = None
    for i in range(n + 1):
        tol = lo * (hi / lo) ** (i / n)
        r_uni = uniform_feasible_rate(T_grid, tau0, k, rate_max, rate_grad_max, tol)
        if r_uni is None:
            continue
        gt = total_time(slew_limit([r_uni] * M, rate_max, rate_grad_max), T_grid)
        bb = band_best(T_grid, tau0, k, rate_max, rate_grad_max, tol, WIDTHS)
        if bb is None:
            continue
        st = bb[0]
        ratio = gt / st
        d = abs(ratio - target_ratio)
        if best is None or d < best[0]:
            best = (d, tol, ratio, gt, st, bb)
    return best


def calibrate_divisor_for_gr(T_grid, tau0, k, rate_max, rate_grad_max, tol, greedy_t,
                              target_gr, iters=50):
    """Find div>1 such that the checker's own reference schedule (uniform
    rate, bisected against the STRICTER tolerance tol/div) has total time
    B with B/(10*greedy_t) == target_gr."""
    M = len(T_grid) - 1
    target_B = target_gr * 10.0 * greedy_t

    def B_at(div):
        r = uniform_feasible_rate(T_grid, tau0, k, rate_max, rate_grad_max, tol / div)
        if r is None:
            return None
        return total_time(slew_limit([r] * M, rate_max, rate_grad_max), T_grid)

    lo, hi = 1.0, 2.0
    for _ in range(60):
        b = B_at(hi)
        if b is not None and b >= target_B:
            break
        hi *= 1.5
    else:
        return None
    for _ in range(iters):
        mid = (lo + hi) / 2
        b = B_at(mid)
        if b is None or b < target_B:
            lo = mid
        else:
            hi = mid
    return hi, B_at(hi)


# ---------- per-testId physical parameter table ----------
# (M, T_hot, T_cold, rate_max, rate_grad_max, frac, C, tau0, target_ratio, target_gr)
PLAN = {
    1:  (22, 880,  300, 240, 42, 0.12, 4.5, 1.1e-4, 3.3, 0.18),
    2:  (26, 920,  300, 260, 48, 0.50, 5.5, 9.0e-5, 3.6, 0.20),
    3:  (24, 900,  300, 250, 45, 0.85, 5.0, 1.0e-4, 3.9, 0.22),
    4:  (30, 1200, 350, 300, 50, 0.30, 4.0, 5.0e-5, 3.4, 0.19),
    5:  (30, 1200, 350, 300, 50, 0.70, 4.2, 4.0e-5, 4.0, 0.21),
    6:  (18, 700,  280, 180, 35, 0.40, 6.0, 2.0e-4, 3.2, 0.17),
    7:  (32, 1000, 300, 260, 40, 0.60, 5.0, 1.0e-4, 3.7, 0.20),
    8:  (20, 800,  320, 200, 32, 0.20, 7.0, 8.0e-5, 3.5, 0.23),
    9:  (36, 1100, 310, 280, 44, 0.50, 5.0, 6.0e-5, 4.1, 0.18),
    10: (26, 950,  300, 240, 38, 0.90, 4.5, 1.2e-4, 3.3, 0.21),
}


def build_instance(testId):
    (M, T_hot, T_cold, rate_max, rate_grad_max, frac, C, tau0,
     target_ratio, target_gr) = PLAN[testId]
    T_grid = build_grid(M, float(T_hot), float(T_cold))
    seg_dur = (T_hot - T_cold) / M / rate_max
    tau_star = C * seg_dur
    Tstar = T_cold + frac * (T_hot - T_cold)
    k = Tstar * math.log(tau_star / tau0)

    res = calibrate_tol_for_ratio(T_grid, tau0, k, rate_max, rate_grad_max, target_ratio)
    if res is None:
        raise RuntimeError(f"testId {testId}: tol calibration failed")
    _, tol, ratio, gt, st, bb = res

    r2 = calibrate_divisor_for_gr(T_grid, tau0, k, rate_max, rate_grad_max, tol, gt, target_gr)
    if r2 is None:
        raise RuntimeError(f"testId {testId}: divisor calibration failed")
    div, B = r2

    return M, T_hot, T_cold, tau0, k, rate_max, rate_grad_max, tol, div


def main():
    testId = int(sys.argv[1])
    M, T_hot, T_cold, tau0, k, rate_max, rate_grad_max, tol, div = build_instance(testId)

    out = []
    out.append(str(M))
    out.append(f"{T_hot} {T_cold}")
    out.append(f"{tau0!r} {k!r}")
    out.append(f"{rate_max} {rate_grad_max}")
    out.append(f"{tol!r} {div!r}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
