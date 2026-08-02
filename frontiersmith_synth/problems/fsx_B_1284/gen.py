import sys
import random
import math

# ESG-screen-portfolio: exclusion-constraint + factor-exposure-drift + substitution-availability.
# Deterministic in testId only.

S = 4          # sectors (categorical factors)
K = S + 1      # + 1 continuous "size" factor

# difficulty ladder: number of benchmark names grows small -> larger.
N_LADDER = [16, 18, 22, 26, 30, 36, 42, 50, 58, 68]

# trap cases: exclusion threshold correlates with a single "dirty" sector, so removing
# the worst ESG names concentrates the exclusion in one sector and tilts that sector's
# factor exposure. testIds 1,3,6,8,10 are factor-neutral controls (exclusion spread evenly).
TRAP_IDS = {2, 4, 5, 7, 9}

T_LADDER = [30.0, 32.0, 35.0, 33.0, 36.0, 34.0, 37.0, 35.0, 38.0, 35.0]


def build_factor_matrix(rng, k, scaleA=0.34, floor=0.010):
    A = [[rng.uniform(-1.0, 1.0) * scaleA for _ in range(k)] for _ in range(k)]
    F = [[0.0] * k for _ in range(k)]
    for a in range(k):
        for b in range(k):
            s = 0.0
            for c in range(k):
                s += A[c][a] * A[c][b]
            F[a][b] = s / k
    for a in range(k):
        F[a][a] += floor
    return F


def main():
    testId = int(sys.argv[1])
    idx = min(max(testId, 1), len(N_LADDER)) - 1
    N = N_LADDER[idx]
    T = T_LADDER[idx]
    trap = testId in TRAP_IDS

    rng = random.Random(1000003 * testId + 17)

    dirty_sector = testId % S

    # raw (unnormalized) lognormal benchmark weights
    raw = [math.exp(rng.gauss(0.0, 0.9)) for _ in range(N)]
    tot = sum(raw)
    w = [r / tot for r in raw]

    # Balanced-but-shuffled sector assignment: round-robin slots so every sector gets
    # (close to) N/S members regardless of testId luck, then shuffle deterministically.
    # (A pure rng.randrange(S) per name can leave a sector with 0-1 members by chance,
    # which would make the "concentrated exclusion" trap toothless.)
    slots = [i % S for i in range(N)]
    rng.shuffle(slots)

    sectors = []
    sizes = []
    esg = []
    for i in range(N):
        sec = slots[i]
        sectors.append(sec)
        sizes.append(rng.uniform(-1.0, 1.0))
        if trap:
            if sec == dirty_sector:
                s = rng.uniform(0.0, 40.0)
            else:
                s = rng.uniform(30.0, 100.0)
        else:
            s = rng.uniform(0.0, 100.0)
        esg.append(s)

    if trap:
        # Force a DETERMINISTIC, strong concentration: almost every name in the dirty
        # sector is excluded, but a couple of same-sector substitutes are deliberately
        # kept eligible (so the within-sector-substitution insight has something to work
        # with -- if the whole sector vanished, no strategy could restore its exposure).
        dsec = [i for i in range(N) if sectors[i] == dirty_sector]
        target_survivors = min(2, len(dsec))
        cur_survivors = [i for i in dsec if esg[i] >= T]
        cur_excluded = [i for i in dsec if esg[i] < T]
        # promote dirty-sector names (in index order, deterministic) until we have
        # exactly target_survivors, then demote everyone else in the sector
        while len(cur_survivors) < target_survivors and cur_excluded:
            i = cur_excluded.pop(0)
            esg[i] = T + rng.uniform(2.0, 15.0)
            cur_survivors.append(i)
        keep = set(cur_survivors[:target_survivors])
        for i in dsec:
            if i not in keep:
                esg[i] = T - rng.uniform(1.0, 12.0)

    # substitution-availability caps: each name can absorb its own benchmark weight
    # scaled up, PLUS a flat liquidity floor independent of its own (possibly tiny)
    # weight -- representing generic tradeable float in a same-sector substitute. The
    # floor is what makes real substitution possible even when a sector's exclusions
    # hit its larger names hardest.
    mult = [rng.uniform(2.0, 5.0) for _ in range(N)]
    floor = [rng.uniform(0.05, 0.14) for _ in range(N)]
    cap = [w[i] * mult[i] + floor[i] for i in range(N)]

    # feasibility safety net: total eligible capacity must exceed 1.0 with margin;
    # bump multipliers uniformly if a pathological draw falls short (should not trigger
    # in practice given the ranges above, but keeps the instance always solvable).
    def eligible_cap_sum():
        return sum(cap[i] for i in range(N) if esg[i] >= T)

    guard = 0
    while eligible_cap_sum() < 1.25 and guard < 200:
        for i in range(N):
            cap[i] *= 1.15
        guard += 1

    d = [rng.uniform(0.0005, 0.0015) for _ in range(N)]

    F = build_factor_matrix(rng, K)

    out = []
    out.append("%d %d" % (N, S))
    out.append("%.6f" % T)
    for row in F:
        out.append(" ".join("%.8f" % v for v in row))
    for i in range(N):
        out.append("%d %.8f %.6f %.10f %.10f %.8f" %
                    (sectors[i], sizes[i], esg[i], w[i], cap[i], d[i]))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
