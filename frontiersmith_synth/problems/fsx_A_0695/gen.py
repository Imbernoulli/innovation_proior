#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE reservoir-scheduling instance to stdout.

Family: flood-margin-release-ladder. One release calendar r_1..r_T (weeks) must stay
feasible (dead-pool/overflow) under EVERY one of K inflow scenarios simultaneously, and
the objective rewards the worst (min) scenario's total power.

Determinism: all randomness is seeded from testId only.

Output:
  line 1: T K C D S0 Rmax HMIN HMAX
  next K lines: T whitespace-separated integers each -- scenario k's weekly inflow
    (week 1..T), k = 0..K-1 in that order.
"""
import random
import math
import sys


def build(test_id: int):
    rnd = random.Random(1000003 * test_id + 7)
    T = [16, 18, 20, 22, 26, 30, 34, 38, 44, 52][(test_id - 1) % 10]
    K = [5, 5, 6, 6, 7, 7, 8, 8, 9, 9][(test_id - 1) % 10]
    calm = test_id <= 3          # near-zero scenario dispersion: any reasonable schedule works
    mild_storm = test_id in (4, 5)   # ONE flood scenario: a single-scenario plan still survives
    # test_id in 6..10: severe storm -- TWO floods peaking at different weeks (the trap)

    base = []
    for t in range(1, T + 1):
        seasonal = 40 + 25 * math.sin(2 * math.pi * (t / T))
        base.append(max(2, int(round(seasonal + rnd.uniform(-6, 6)))))

    scenarios = []
    if calm:
        for _ in range(K):
            scenarios.append(list(base))
    else:
        nbulk = max(2, K - (2 if mild_storm else 3))
        for _ in range(nbulk):
            scenarios.append([max(1, int(round(v * rnd.uniform(0.95, 1.05)))) for v in base])
        scenarios.append([max(1, int(round(v * 0.5))) for v in base])   # drought scenario

        sumb = sum(base)

        def pulse(frac_center, width_frac, total_frac):
            center = max(1, int(round(T * frac_center)))
            width = max(2, int(T * width_frac))
            w = [math.exp(-((t - center) ** 2) / (2 * (width ** 2))) for t in range(1, T + 1)]
            sw = sum(w)
            total = int(sumb * total_frac * rnd.uniform(0.95, 1.05))
            sc = list(base)
            rem = total
            for i in range(T):
                add = min(int(round(total * w[i] / sw)), rem)
                sc[i] += add
                rem -= add
            return sc

        if mild_storm:
            scenarios.append(pulse(0.50, 1.0 / 6, 0.85))
        else:
            scenarios.append(pulse(0.20, 1.0 / 14, 0.50))   # early, smaller-total flood
            scenarios.append(pulse(0.78, 1.0 / 6, 0.95))    # late, bigger-total flood

    while len(scenarios) < K:
        scenarios.append([max(1, int(round(v * rnd.uniform(0.95, 1.05)))) for v in base])
    scenarios = scenarios[:K]

    # running prefix max/min over scenarios of cumulative inflow (the envelope)
    cum = [[0] * (T + 1) for _ in range(K)]
    for k in range(K):
        for t in range(1, T + 1):
            cum[k][t] = cum[k][t - 1] + scenarios[k][t - 1]
    max_in = [max(cum[k][t] for k in range(K)) for t in range(T + 1)]
    min_in = [min(cum[k][t] for k in range(K)) for t in range(T + 1)]

    spread = max(max_in[t] - min_in[t] for t in range(T + 1))
    buffer = int(math.ceil(spread * 1.25)) + 25
    D = int(round(buffer * 0.28))
    C = D + buffer
    S0 = D + buffer // 2

    max_week_inflow = max(max(sc) for sc in scenarios)
    rel_lo = [0] * (T + 1)
    for t in range(1, T + 1):
        rel_lo[t] = max(rel_lo[t - 1], max(0, S0 + max_in[t] - C))
    inc_lo = max(rel_lo[t] - rel_lo[t - 1] for t in range(1, T + 1))
    Rmax = max(2 * max_week_inflow, 2 * inc_lo, 10)

    HMIN, HMAX = 100, 1000   # per-mille head multiplier bounds (0.1 .. 1.0)

    return T, K, C, D, S0, Rmax, HMIN, HMAX, scenarios


def main():
    test_id = int(sys.argv[1])
    T, K, C, D, S0, Rmax, HMIN, HMAX, scenarios = build(test_id)
    out = [f"{T} {K} {C} {D} {S0} {Rmax} {HMIN} {HMAX}"]
    for k in range(K):
        out.append(" ".join(str(x) for x in scenarios[k]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
