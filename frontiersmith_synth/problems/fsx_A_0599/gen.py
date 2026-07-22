import sys, random

# Distinct E24-style resistor values (ohms, one decade). gen takes the first M.
CATALOG = [100, 110, 120, 130, 150, 160, 180, 200, 220, 240, 270, 300, 330, 360]

TPM = 50            # 5 % tolerance, per-mille
NCORNERS = 128

# "Easy" near-1/2 target ratios: each is exactly a low-denominator fraction b/(a+b)
# with a+b <= 15, so a matched-batch (single-value) build hits it with residual 0.
# 3/7 and 4/7 are the farthest from 1/2 (dist 0.0714) and pin the checker baseline B.
EASY = [3/7, 4/7, 5/9, 4/9, 6/11, 5/11, 7/13, 6/13, 8/15, 7/15]

# "Hard" offsets from 1/2: 1/2 is the nearest matched ratio for a long stretch, so a
# target 1/2+delta forces residual == delta until denominators ~40+ are spent. This
# planted tap sets the strong-tier worst-corner error (and leaves headroom above it).
HARD_DELTAS = [0.0085, 0.0095, 0.0105]


def gen_corners(rng, M, C):
    seen, corners, tries = set(), [], 0
    while len(corners) < C and tries < 60 * C:
        tries += 1
        v = tuple(rng.choice((-1, 1)) for _ in range(M))
        if len(set(v)) == 1:            # uniform corner scales all parts equally -> zero deviation
            continue
        if v in seen:
            continue
        seen.add(v)
        corners.append(v)
    while len(corners) < C:              # pad if the unique pool ran dry (small M)
        v = tuple(rng.choice((-1, 1)) for _ in range(M))
        if len(set(v)) != 1:
            corners.append(v)
    return corners


def main():
    tid = int(sys.argv[1])
    rng = random.Random(45990000 + 101 * tid)

    if tid <= 3:
        K, M = 4, 6
    elif tid <= 6:
        K, M = 6, 8
    elif tid <= 8:
        K, M = 7, 10
    else:
        K, M = 8, 12

    catalog = CATALOG[:M]
    P = 32 * K
    C = NCORNERS if tid > 2 else 64

    # targets: always 3/7 and 4/7 (pin B), one planted hard tap, rest easy fractions.
    delta = HARD_DELTAS[tid % len(HARD_DELTAS)]
    hard = 0.5 + (delta if (tid % 2 == 0) else -delta)
    pool = EASY[2:]
    rng.shuffle(pool)
    targets = [EASY[0], EASY[1], hard]
    i = 0
    while len(targets) < K:
        targets.append(pool[i % len(pool)])
        i += 1
    rng.shuffle(targets)

    corners = gen_corners(rng, M, C)

    out = ["%d %d %d %d %d" % (K, M, P, C, TPM),
           " ".join("%.6f" % r for r in targets),
           " ".join(str(v) for v in catalog)]
    for cv in corners:
        out.append(" ".join(str(s) for s in cv))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
