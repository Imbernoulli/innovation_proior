#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- checker for glass-anneal-schedule.

Reads the instance (segment count M, temperature endpoints, Arrhenius
relaxation-time law, oven rate caps, mismatch tolerance, baseline
strictness divisor) from <in>. Reads the participant's cooling schedule
(M positive durations t_1..t_M, hottest segment first) from <out>.

Feasibility (any violation -> `Ratio: 0.0`):
  - exactly M whitespace-separated tokens, each a finite positive number.
  - the implied rate r_i = dT_i/t_i must satisfy 0 < r_i <= rate_max.
  - |r_i - r_{i-1}| <= rate_grad_max for every i (r_0 := 0: the oven
    starts at rest, so the first segment's rate is also gradient-capped).
  - the final structural mismatch |Tf_M - T_cold| (fictive vs. actual
    temperature, via the closed-form per-segment recurrence below) must
    be <= tol.

Objective (to MINIMIZE): F = sum(t_i), the total anneal time.

The checker's own reference is a uniform-rate schedule (same slew-limited
bisection a solver could run) but found against a STRICTER effective
tolerance tol/div -- i.e. an extra-cautious "just go uniformly slow"
construction, giving total time B. Score:
  Ratio = min(1, 100*B/F) / 10   (a 10x-faster schedule saturates at 1.0)
"""
import sys, math

EPS_REL = 1e-6


def fail(msg):
    print(msg)
    print("Ratio: 0.0")
    sys.exit(0)


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


def uniform_feasible_rate(T_grid, tau0, k, rate_max, rate_grad_max, tol, iters=60):
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


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    M = int(next(it))
    T_hot = float(next(it)); T_cold = float(next(it))
    tau0 = float(next(it)); k = float(next(it))
    rate_max = float(next(it)); rate_grad_max = float(next(it))
    tol = float(next(it)); div = float(next(it))
    return M, T_hot, T_cold, tau0, k, rate_max, rate_grad_max, tol, div


def main():
    if len(sys.argv) < 3:
        fail("bad args")
    in_path, out_path = sys.argv[1], sys.argv[2]
    M, T_hot, T_cold, tau0, k, rate_max, rate_grad_max, tol, div = read_instance(in_path)
    T_grid = build_grid(M, T_hot, T_cold)

    with open(out_path) as f:
        raw = f.read().split()

    if len(raw) != M:
        fail(f"expected {M} tokens, got {len(raw)}")

    t = []
    for tok in raw:
        try:
            v = float(tok)
        except ValueError:
            fail(f"non-numeric token: {tok!r}")
        if not math.isfinite(v):
            fail("non-finite token")
        if v <= 0.0:
            fail(f"non-positive duration: {v}")
        t.append(v)

    rate_tol = rate_max * EPS_REL + 1e-9
    grad_tol = rate_grad_max * EPS_REL + 1e-9
    rates = []
    prev_r = 0.0
    for i in range(M):
        dT = T_grid[i] - T_grid[i + 1]
        r = dT / t[i]
        if r > rate_max + rate_tol:
            fail(f"segment {i+1}: rate {r:.6f} exceeds rate_max {rate_max}")
        if abs(r - prev_r) > rate_grad_max + grad_tol:
            fail(f"segment {i+1}: rate jump {abs(r-prev_r):.6f} exceeds rate_grad_max {rate_grad_max}")
        rates.append(r)
        prev_r = r

    mismatch = simulate_mismatch(rates, T_grid, tau0, k)
    if not math.isfinite(mismatch):
        fail("non-finite mismatch")
    if mismatch > tol + tol * EPS_REL + 1e-9:
        fail(f"residual mismatch {mismatch:.6f} exceeds tolerance {tol:.6f}")

    F = sum(t)
    if not math.isfinite(F) or F <= 0.0:
        fail("invalid total time")

    r_ref = uniform_feasible_rate(T_grid, tau0, k, rate_max, rate_grad_max, tol / div)
    if r_ref is None:
        fail("internal: reference schedule infeasible")
    B = total_time(slew_limit([r_ref] * M, rate_max, rate_grad_max), T_grid)
    B = max(1e-9, B)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    print(f"total_time(F)={F:.6f} baseline(B)={B:.6f} mismatch={mismatch:.6f} Ratio: {ratio:.6f}")


if __name__ == "__main__":
    main()
