#!/usr/bin/env python3
"""gen.py <testId> -> prints ONE instance of 'word-autocorrelation-realize' to stdout.

Instance:
  line 1: n K lam alpha
  line 2: m
  next m lines: b_i w_i   (target border length, weight)

Seeded purely by testId (deterministic, reproducible).
"""
import sys, math, random


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def forced_periods(n, base_periods):
    """Ground-truth closure: union positions i~i+p for every base period p (for
    all valid i), then report every q in [1,n-1] whose equality W[i]=W[i+q] is
    ALREADY implied by that connectivity. This captures both 'multiples of a
    single period are periods too' and the Fine-and-Wilf gcd-forcing rule (and
    any deeper multi-way interaction) via one mechanical construction -- no
    separate case analysis needed."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for p in base_periods:
        for i in range(n - p):
            union(i, i + p)

    forced = set()
    for q in range(1, n):
        if all(find(i) == find(i + q) for i in range(n - q)):
            forced.add(q)
    return forced


def num_classes(n, base_periods):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for p in base_periods:
        for i in range(n - p):
            union(i, i + p)
    return len({find(i) for i in range(n)})


def make_trap_periods(rng, n):
    """Two base periods, moderate in size, chosen so that realizing BOTH as
    targets mathematically forces a SMALL but nonzero number of extra periods
    (>=1, <=4) via forced_periods -- a controlled, non-catastrophic trap that
    still rewards recognizing the interaction instead of naive independent
    stitching."""
    best = None
    for _ in range(600):
        p1 = rng.randint(max(3, n // 4), n // 2)
        span = max(2, n // 3)
        p2 = rng.randint(p1 + 1, min(n - 2, p1 + span))
        if p2 <= p1:
            continue
        base = sorted({p1, p2})
        if len(base) < 2:
            continue
        fp = forced_periods(n, base)
        extra = fp - set(base)
        if 1 <= len(extra) <= 4:
            return base
        if best is None:
            best = base
    return best if best is not None else sorted({n // 3, n // 2})


def make_free_periods(rng, n):
    """Periods close to n (short borders), spread out so essentially no extra
    period is forced -- the well-behaved (non-trap) regime."""
    lo = max(n // 2 + 1, 3)
    hi = n - 2
    if hi <= lo:
        return [n - 2]
    cand = list(range(lo, hi + 1))
    rng.shuffle(cand)
    chosen = []
    target_count = rng.choice([2, 2, 3])
    for p in cand:
        base_try = chosen + [p]
        fp = forced_periods(n, base_try)
        if len(fp - set(base_try)) == 0:
            chosen.append(p)
        if len(chosen) >= target_count:
            break
    if not chosen:
        chosen = [hi]
    return sorted(chosen)


def main():
    testId = int(sys.argv[1])
    rng = random.Random(20260 + 97 * testId)

    n = 18 + 3 * testId          # 21 .. 48
    trap = testId in (2, 3, 5, 7, 9, 10)  # >=3 of the 10 are trap cases (we use 6)

    if trap:
        periods = make_trap_periods(rng, n)
    else:
        periods = make_free_periods(rng, n)

    nc = num_classes(n, periods)
    K = max(3, min(12, nc + rng.randint(2, 4)))

    lam = 1 + (testId % 2)          # 1 or 2 (kept modest so forced extras don't blow the budget)
    alpha = 1 + (testId % 3 == 0)   # 1 or 2

    borders = sorted(n - p for p in periods)
    weights = [rng.randint(3, 5) for _ in borders]
    items = list(zip(borders, weights))
    rng.shuffle(items)  # target order in the input is NOT sorted (naive stitchers
                         # that process "as given" get no free help from ordering)

    out = []
    out.append(f"{n} {K} {lam} {alpha}")
    out.append(str(len(items)))
    for b, w in items:
        out.append(f"{b} {w}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
