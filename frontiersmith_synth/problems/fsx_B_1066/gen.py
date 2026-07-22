#!/usr/bin/env python3
"""gen.py <testId> -- emits one instance of "Rainproof Season" to stdout.

Deterministic: all randomness is seeded from testId only.
"""
import sys
import random


def canonical_pairs(N):
    return [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]


def first_fit_schedule(N, D, C):
    """Naive first-fit edge-coloring construction (lexicographic pair order,
    earliest legal (date,court) slot). Returns list[(date,court)] indexed by
    canonical game order, or None if it does not fit within D dates."""
    busy = [set() for _ in range(D + 1)]
    ccount = [0] * (D + 1)
    sched = []
    for (i, j) in canonical_pairs(N):
        placed = False
        for d in range(1, D + 1):
            if ccount[d] < C and i not in busy[d] and j not in busy[d]:
                ccount[d] += 1
                busy[d].add(i)
                busy[d].add(j)
                sched.append((d, ccount[d]))
                placed = True
                break
        if not placed:
            return None
    return sched


def first_fit_dates_needed(N, C, cap_mult=6):
    d = first_fit_schedule(N, N * cap_mult, C)
    if d is None:
        return None
    return max(x[0] for x in d)


# ---- per-testId difficulty ladder ----
TABLE = {
    1:  dict(N=6,  bonus=5, K=5),
    2:  dict(N=8,  bonus=6, K=6),
    3:  dict(N=8,  bonus=7, K=6),
    4:  dict(N=10, bonus=7, K=7),
    5:  dict(N=10, bonus=8, K=7),
    6:  dict(N=12, bonus=8, K=8),
    7:  dict(N=12, bonus=9, K=8),
    8:  dict(N=14, bonus=9, K=9),
    9:  dict(N=14, bonus=10, K=9),
    10: dict(N=16, bonus=10, K=10),
}


def _gen_one_scenario(rng, minimal, kind):
    if kind == 0:
        d = rng.randint(1, minimal)
        return [d]
    elif kind in (1, 2):
        lo = max(2, minimal // 3)
        hi = max(lo, (2 * minimal) // 3)
        size = rng.randint(lo, hi)
        size = min(size, minimal)
        start = rng.randint(1, max(1, minimal - size + 1))
        return list(range(start, start + size))
    else:
        half = max(1, minimal // 2)
        size1 = rng.randint(1, max(1, half // 2))
        size2 = rng.randint(1, max(1, (minimal - half) // 2))
        size1 = min(size1, half)
        size2 = min(size2, max(1, minimal - half))
        start1 = rng.randint(1, max(1, half - size1 + 1))
        lo2 = half + 1
        hi2 = max(lo2, minimal - size2 + 1)
        start2 = rng.randint(lo2, hi2) if hi2 >= lo2 else lo2
        block = sorted(set(range(start1, start1 + size1)) |
                        set(range(start2, min(minimal, start2 + size2 - 1) + 1)))
        return block


def gen_scenarios(rng, minimal, K):
    """Mix of isolated single-date cancellations, large contiguous cluster
    cancellations (the primary trap: a whole run of rounds wiped at once),
    and double-cluster cancellations (two disjoint runs, forcing coverage
    of more than one region -- a single well-placed bye cannot help both).
    Each scenario is kept distinct from every earlier one in this test (a
    duplicate scenario is scored-away for free under the max-over-scenarios
    objective, so it would silently shrink the effective K)."""
    scenarios = []
    seen = set()
    for k in range(K):
        kind = k % 4
        block = _gen_one_scenario(rng, minimal, kind)
        tries = 0
        while tuple(block) in seen and tries < 50:
            block = _gen_one_scenario(rng, minimal, kind)
            tries += 1
        seen.add(tuple(block))
        scenarios.append(block)
    return scenarios


def main():
    testId = int(sys.argv[1])
    cfg = TABLE[testId]
    N = cfg["N"]
    C = N // 2
    minimal = N - 1
    need = first_fit_dates_needed(N, C)
    D = max(minimal, need) + cfg["bonus"]
    K = cfg["K"]
    R = 2
    LAM = 3

    rng = random.Random(20000 + 131 * testId + N)
    scenarios = gen_scenarios(rng, minimal, K)

    out = []
    out.append(f"{N} {D} {C} {K} {R} {LAM}")
    for sc in scenarios:
        out.append(f"{len(sc)} " + " ".join(str(x) for x in sc))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
