#!/usr/bin/env python3
"""gen.py <testId> -- print ONE royalty-audit-sample instance to stdout (format C).

Population of N transactions split into K=3 strata:
  0 Marquee : few, very-high-value, expensive-to-audit accounts (already well controlled)
  1 Standard: mid-count, mid-value accounts
  2 LongTail: many, small-value, cheap-to-audit accounts (the highest true error rate)

STDOUT is PUBLIC data only: population sizes, reported values, per-transaction audit
cost, a rough historical PRIOR estimate of each stratum's error rate (mean, stdev), the
audit budget, and this test's extrapolation-defensibility threshold. The ACTUAL error
found upon audit ("hidden truth") is never printed here -- it lives only inside verify.py,
freshly regenerated per testId, so a submitted plan cannot special-case it.

Difficulty ladder (testId 1..N): larger testId => bigger population, tighter
extrapolation-defensibility threshold (thresh shrinks), so a defensible claim needs
a more careful stratified design.
"""
import sys, math, random

# (mean_rate, cv, val_lo, val_hi, cost_frac, cost_fixed_lo, cost_fixed_hi, pop_frac)
STRATA = [
    (0.020, 0.65, 40000, 140000, 0.010, 60, 140, 0.14),   # 0 Marquee
    (0.055, 0.60,  3000,  18000, 0.020, 15,  40, 0.34),   # 1 Standard
    (0.110, 0.55,   150,  2200,  0.035,  3,  12, 0.52),   # 2 LongTail
]
Z = 1.645
MIN_STRATUM = 8


def needed_n(cv, thresh):
    return math.ceil((Z * cv / thresh) ** 2)


def systematic_pick_cost(members_sorted_by_value, target):
    """Cost of a value-spread systematic sample of `target` items (cheapest item in
    each of `target` equal-width value slices) -- used only to size the budget."""
    nb = len(members_sorted_by_value)
    if target <= 0 or nb == 0:
        return 0
    step = nb / target
    total = 0
    for k in range(target):
        i = min(nb - 1, int(k * step + step / 2))
        total += members_sorted_by_value[i][3]  # cost is field index 3
    return total


def build_population(t):
    rng = random.Random(20000 + 97 * t)
    N = 65 + 6 * t
    counts = [max(MIN_STRATUM, round(N * STRATA[h][7])) for h in range(3)]
    counts[-1] += N - sum(counts)
    counts[-1] = max(MIN_STRATUM, counts[-1])
    N = sum(counts)

    rows = []  # (id, stratum, value, cost)
    tid = 1
    for h, c in enumerate(counts):
        _, _, vlo, vhi, cf, clo, chi, _ = STRATA[h]
        for _ in range(c):
            v = rng.randint(vlo, vhi)
            cost = int(round(cf * v)) + rng.randint(clo, chi)
            rows.append([tid, h, v, cost])
            tid += 1

    thresh = max(0.30, 0.45 - 0.012 * t)

    by_h = {h: [r for r in rows if r[1] == h] for h in range(3)}
    cost_marquee_full = sum(r[3] for r in by_h[0])

    n0_std = needed_n(STRATA[1][1], thresh)
    n0_lt = needed_n(STRATA[2][1], thresh)
    target_std = min(len(by_h[1]), max(n0_std, 6) + 3)
    target_lt = min(len(by_h[2]), max(n0_lt, 6) + 3)

    std_by_val = sorted(by_h[1], key=lambda r: r[2])
    lt_by_val = sorted(by_h[2], key=lambda r: r[2])
    cost_std_target = systematic_pick_cost(std_by_val, target_std)
    cost_lt_target = systematic_pick_cost(lt_by_val, target_lt)

    slack = 1.08 - 0.004 * t
    Cmax = int(round(cost_marquee_full + slack * (cost_std_target + cost_lt_target)))

    priors = []
    for h in range(3):
        mean, cv = STRATA[h][0], STRATA[h][1]
        jrng = random.Random(5000 + 13 * t + h)
        mean_j = mean * (1 + jrng.uniform(-0.10, 0.10))
        sd_j = mean * cv * (1 + jrng.uniform(-0.10, 0.10))
        priors.append((mean_j, sd_j))

    return N, Cmax, thresh, priors, rows


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        return 1
    t = int(sys.argv[1])
    if t < 1:
        t = 1

    N, Cmax, thresh, priors, rows = build_population(t)

    out = []
    out.append("%d %d %d %d" % (t, N, 3, Cmax))
    out.append("%.6f" % thresh)
    for mean_j, sd_j in priors:
        out.append("%.6f %.6f" % (mean_j, sd_j))
    for tid, h, v, cost in rows:
        out.append("%d %d %d %d" % (tid, h, v, cost))

    sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
