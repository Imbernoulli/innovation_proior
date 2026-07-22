#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -> prints 'Ratio: <x in [0,1]>' (last line authoritative).

Deterministic scorer for the dam release-calendar problem.

Instance (<in>):
  line 1: T K C D S0 Rmax HMIN HMAX
  next K lines: T integers each -- scenario k's weekly inflow

Participant output (<out>): exactly T whitespace-separated integers r_1..r_T (the SAME
release calendar simulated against every one of the K scenarios).

Feasibility (checked for EVERY scenario k and EVERY week t):
  0 <= r_t <= Rmax                      (design release capacity)
  D <= S_t^k <= C                       (dead pool / overflow, hard)
where S_0^k = S0 and S_t^k = S_{t-1}^k + inflow_t^k - r_t.
ANY violation, in ANY scenario -> Ratio 0.0.

Objective: F = min_k  sum_t  shape(r_t) * head_mult(S_{t-1}^k)
  shape(r, Rmax)      = r - r^2 / (2.2 * Rmax)                 (concave in release)
  head_mult(S, C, D)  = HMIN/1000 + (HMAX-HMIN)/1000 * (2x - x^2), x = clip((S-D)/(C-D), 0, 1)
                                                                  (concave increasing in storage)

Baseline B (checker-internal, always feasible): a "burst then wait" schedule that, whenever
forced, jumps the release as far as the weekly cap and the dead-pool bound allow, then holds
until forced again. It satisfies the SAME true worst-case corridor as an optimal schedule but
ignores the concavity of shape()/head_mult(), so it scores well below the informed optimum.

Normalization (maximization): sc = min(1000, 100*F/max(1e-9,B)); Ratio = sc/1000.
"""
import sys


def fail(reason):
    sys.stdout.write("reason: %s\nRatio: 0.0\n" % reason)
    sys.exit(0)


def read_instance(path):
    try:
        with open(path) as f:
            toks = f.read().split()
    except Exception:
        fail("cannot read instance")
    it = iter(toks)
    try:
        T = int(next(it)); K = int(next(it)); C = int(next(it)); D = int(next(it))
        S0 = int(next(it)); Rmax = int(next(it)); HMIN = int(next(it)); HMAX = int(next(it))
        scenarios = []
        for _ in range(K):
            sc = [int(next(it)) for _ in range(T)]
            scenarios.append(sc)
    except (StopIteration, ValueError):
        fail("malformed instance")
    return T, K, C, D, S0, Rmax, HMIN, HMAX, scenarios


def envelope(T, K, S0, C, D, scenarios):
    """running prefix max/min over scenarios of cumulative inflow, and the resulting
    hard feasibility corridor [lo(t), hi(t)] for the CUMULATIVE release Rel(t)."""
    cum = [[0] * (T + 1) for _ in range(K)]
    for k in range(K):
        for t in range(1, T + 1):
            cum[k][t] = cum[k][t - 1] + scenarios[k][t - 1]
    max_in = [max(cum[k][t] for k in range(K)) for t in range(T + 1)]
    min_in = [min(cum[k][t] for k in range(K)) for t in range(T + 1)]
    lo = [0] * (T + 1)
    hi = [0] * (T + 1)
    for t in range(1, T + 1):
        lo[t] = max(lo[t - 1], max(0, S0 + max_in[t] - C))
        hi[t] = max(hi[t - 1], S0 + min_in[t] - D)
    return lo, hi


def baseline_release(T, K, S0, C, D, Rmax, scenarios):
    lo, hi = envelope(T, K, S0, C, D, scenarios)
    Rel = [0] * (T + 1)
    for t in range(1, T + 1):
        if Rel[t - 1] < lo[t]:
            Rel[t] = min(hi[t], max(lo[t], Rel[t - 1] + Rmax))
        else:
            Rel[t] = Rel[t - 1]
    return [Rel[t] - Rel[t - 1] for t in range(1, T + 1)]


def head_mult(S, C, D, HMIN, HMAX):
    x = (S - D) / (C - D)
    x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
    return (HMIN + (HMAX - HMIN) * (2 * x - x * x)) / 1000.0


def shape(r, Rmax):
    r = 0.0 if r < 0 else (Rmax if r > Rmax else r)
    return r - (r * r) / (2.2 * Rmax)


def simulate_and_score(T, K, C, D, S0, Rmax, HMIN, HMAX, scenarios, r):
    """Returns (F, ok, reason). F is min-over-scenarios total power (only meaningful if ok)."""
    totals = []
    for k in range(K):
        s = S0
        storages = [s]
        for t in range(1, T + 1):
            s = s + scenarios[k][t - 1] - r[t - 1]
            if s < D - 1e-9 or s > C + 1e-9:
                return None, False, f"scenario {k} week {t}: storage {s} outside [{D},{C}]"
            storages.append(s)
        tot = 0.0
        for t in range(1, T + 1):
            tot += shape(r[t - 1], Rmax) * head_mult(storages[t - 1], C, D, HMIN, HMAX)
        totals.append(tot)
    return min(totals), True, ""


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    T, K, C, D, S0, Rmax, HMIN, HMAX, scenarios = read_instance(inp)

    try:
        with open(outp) as f:
            raw = f.read().split()
    except Exception:
        fail("cannot read output")

    if len(raw) != T:
        fail("expected exactly %d tokens, got %d" % (T, len(raw)))

    r = []
    for tok in raw:
        try:
            v = int(tok)
        except ValueError:
            try:
                fv = float(tok)
            except ValueError:
                fail("non-numeric token %r" % tok)
            if fv != fv or fv in (float("inf"), float("-inf")):
                fail("non-finite token %r" % tok)
            fail("non-integer token %r" % tok)
        if v < 0 or v > Rmax:
            fail("release %d outside [0,%d] at week %d" % (v, Rmax, len(r) + 1))
        r.append(v)

    F, ok, reason = simulate_and_score(T, K, C, D, S0, Rmax, HMIN, HMAX, scenarios, r)
    if not ok:
        fail(reason)

    b_r = baseline_release(T, K, S0, C, D, Rmax, scenarios)
    B, b_ok, b_reason = simulate_and_score(T, K, C, D, S0, Rmax, HMIN, HMAX, scenarios, b_r)
    if not b_ok:
        fail("internal baseline infeasible (%s) -- generator bug" % b_reason)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    ratio = sc / 1000.0
    sys.stdout.write("F=%.6f B=%.6f Ratio: %.6f\n" % (F, B, ratio))
    sys.exit(0)


if __name__ == "__main__":
    main()
