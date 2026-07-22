import sys, random

# Family layout (fixed order, K = 6):
#   0 const_shift      -- whole-segment additive bug            (LARGE region)
#   1 coef_perturb     -- whole-segment slope bug                (LARGE region)
#   2 sign_flip        -- whole-segment sign bug                 (LARGE region)
#   3 boundary_shift   -- single off-by-one point at a breakpoint (TINY region, but at an "obvious" spot)
#   4 interior_anomaly -- narrow window strictly inside a segment's interior (TINY, NOT at a boundary)
#   5 sparse_anomaly   -- single point deep inside a segment's interior     (POINT, NOT at a boundary)
#
# Difficulty ladder over testId 1..10: domain size D and segment count S grow.
KFAM = 6
MOD = 1009
L_MIN = 16


def build_instance(test_id):
    rng = random.Random(1000 + test_id * 97)
    D = 300 + test_id * 140
    S = 6 + (test_id % 4)

    base_len = D // S
    breakpoints = [0]
    for i in range(1, S):
        jitter = rng.randint(-(base_len // 4), base_len // 4)
        bp = i * base_len + jitter
        bp = max(breakpoints[-1] + 5, bp)
        breakpoints.append(bp)
    breakpoints.append(D)

    segs = []
    for i in range(S):
        lo, hi = breakpoints[i], breakpoints[i + 1]
        a = rng.randint(1, 9)
        c = rng.randint(0, MOD - 1)
        segs.append((lo, hi, a, c))

    families = [[] for _ in range(KFAM)]

    for (lo, hi, a, c) in segs:
        families[0].append((lo, hi))
    for (lo, hi, a, c) in segs:
        families[1].append((lo, hi))
    for (lo, hi, a, c) in segs:
        families[2].append((lo, hi))
    for j in range(1, S):
        b = breakpoints[j]
        families[3].append((b, b + 1))
    for (lo, hi, a, c) in segs:
        L = hi - lo
        if L < L_MIN:
            continue
        margin = max(3, L // 4)
        inner_lo, inner_hi = lo + margin, hi - margin
        if inner_hi - inner_lo < 3:
            continue
        w = max(2, (inner_hi - inner_lo) // 3)
        start = rng.randint(inner_lo, inner_hi - w)
        families[4].append((start, start + w))
    for (lo, hi, a, c) in segs:
        L = hi - lo
        if L < L_MIN:
            continue
        margin = max(3, L // 4)
        inner_lo, inner_hi = lo + margin, hi - margin
        if inner_hi <= inner_lo:
            continue
        x0 = rng.randint(inner_lo, inner_hi - 1)
        families[5].append((x0, x0 + 1))

    # every family must be non-empty; if degenerate, widen margins and retry once
    if any(len(f) == 0 for f in families):
        for fi in (4, 5):
            if not families[fi]:
                lo, hi, a, c = segs[0]
                mid = (lo + hi) // 2
                families[fi].append((mid, mid + 1))

    T = S
    return D, T, families


def main():
    test_id = int(sys.argv[1])
    D, T, families = build_instance(test_id)
    out = [f"{D} {T} {KFAM}"]
    for fam in families:
        out.append(str(len(fam)))
        for lo, hi in fam:
            out.append(f"{lo} {hi}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
