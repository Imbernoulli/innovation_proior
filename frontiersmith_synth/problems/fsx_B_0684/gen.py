#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE shoulder-shadow-turning instance to stdout.

Instance format:
  line 1: n m K C R
  line 2: target_1 ... target_n        (1 <= target_i <= R-1)
  next m lines: maxDepth_t minDepth_t  (1 <= minDepth_t <= maxDepth_t)

All randomness is seeded purely from testId (deterministic).
"""
import sys, random


def tools_for(R):
    # maxDepth of every tool scales proportionally with R so the RATIO between
    # tool tiers (and hence the achievable-vs-baseline improvement ceiling)
    # stays roughly scale-invariant instead of blowing up for large R.
    rough_max = max(6, R // 4)
    rough_min = max(3, R // 12)
    if rough_min > rough_max:
        rough_min = rough_max
    med_max = max(4, R // 10)
    med_min = 2
    if med_min > med_max:
        med_min = med_max
    fin_max = max(2, R // 25)
    fin_min = 1
    return [(rough_max, rough_min), (med_max, med_min), (fin_max, fin_min)]


def dip_profile(n, R, dip_left, dip_right, plateau):
    """proximal band(s) AND distal band(s) both need deep removal (small target);
    the interior plateau needs only shallow removal (target close to R)."""
    t = [plateau] * n
    for i in range(dip_left):
        t[i] = 1 + i % 3          # bands 1..dip_left: deep cut needed
    for i in range(dip_right):
        t[n - 1 - i] = 1 + i % 3  # last dip_right bands: deep cut needed
    for i in range(n):
        t[i] = max(1, min(R - 1, t[i]))
    return t


def monotone_profile(n, R, ascending, rng):
    step = max(1, (R - 2) // max(1, n - 1))
    t = []
    for i in range(n):
        v = 1 + i * step if ascending else (R - 1) - i * step
        v += rng.randint(-1, 1)
        t.append(max(1, min(R - 1, v)))
    return t


def multi_dip_profile(n, R, rng, n_dips):
    plateau = R - 1
    t = [plateau] * n
    period = max(2, n // (n_dips + 1))
    for k in range(1, n_dips + 1):
        pos = min(n - 1, k * period)
        t[pos] = 1 + (rng.randint(0, 2))
        if pos + 1 < n:
            t[pos + 1] = max(1, min(R - 1, t[pos + 1] - rng.randint(0, 2)))
    return [max(1, min(R - 1, v)) for v in t]


def random_smooth_profile(n, R, rng):
    v = rng.randint(R // 3, 2 * R // 3)
    t = []
    for _ in range(n):
        v += rng.randint(-max(1, R // 20), max(1, R // 20))
        v = max(1, min(R - 1, v))
        t.append(v)
    return t


CASES = [
    # (n, R, kind)  kind in {"dip","mono_desc","mono_asc","multidip","random","bigdip"}
    (4, 20, "mono_desc"),
    (6, 30, "dip"),
    (8, 40, "mono_asc"),
    (10, 50, "multidip"),
    (15, 60, "bigdip"),
    (20, 80, "random"),
    (25, 100, "bigdip"),
    (35, 120, "mono_desc"),
    (45, 150, "bigdip"),
    (60, 200, "multidip"),
]


def make_case(testId):
    rng = random.Random(1000003 * testId + 17)
    n, R, kind = CASES[(testId - 1) % len(CASES)]
    if kind == "mono_desc":
        target = monotone_profile(n, R, ascending=False, rng=rng)
    elif kind == "mono_asc":
        target = monotone_profile(n, R, ascending=True, rng=rng)
    elif kind == "dip":
        target = dip_profile(n, R, dip_left=1, dip_right=1, plateau=R - 1)
    elif kind == "bigdip":
        left = max(1, n // 6)
        right = max(1, n // 6)
        target = dip_profile(n, R, dip_left=left, dip_right=right, plateau=R - 1)
    elif kind == "multidip":
        target = multi_dip_profile(n, R, rng, n_dips=max(2, n // 8))
    else:  # random
        target = random_smooth_profile(n, R, rng)

    # guarantee bounds
    target = [max(1, min(R - 1, v)) for v in target]

    m = 3
    K = 2 * n
    C = 6
    tools = tools_for(R)

    out = []
    out.append(f"{n} {m} {K} {C} {R}")
    out.append(" ".join(str(v) for v in target))
    for (mx, mn) in tools:
        out.append(f"{mx} {mn}")
    return "\n".join(out) + "\n"


def main():
    testId = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    sys.stdout.write(make_case(testId))


if __name__ == "__main__":
    main()
