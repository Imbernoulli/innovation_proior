import sys, random

# Difficulty ladder: donor-pool size grows small -> large; the last five cases (6..10)
# plant a bimodal donor pool (a concentrated mass of high-warm-glow "regular givers" plus
# a mass of low-warm-glow donors) with a tighter budget -- the inframarginal-crowd-out trap.
SIZES = [6, 10, 16, 25, 40, 60, 90, 140, 200, 320]
TRAP = [False, False, False, False, False, True, True, True, True, True]
K_MAX = 4
R_MAX = 4.0
SEED_BASE = 3000


def gen_donors(rng, N, trap):
    donors = []
    for _ in range(N):
        if trap:
            if rng.random() < 0.55:
                a = rng.uniform(45.0, 75.0)
            else:
                a = rng.uniform(5.0, 15.0)
        else:
            a = rng.uniform(10.0, 30.0)
        w = a * rng.uniform(2.5, 4.5)
        donors.append((a, w))
    return donors


def counterfactual_total(donors):
    tot = 0.0
    for a, w in donors:
        g0 = a - 1.0
        if g0 < 0.0:
            g0 = 0.0
        if g0 > w:
            g0 = w
        tot += g0
    return tot


def main():
    tid = int(sys.argv[1])
    idx = min(max(tid, 1), len(SIZES)) - 1
    N = SIZES[idx]
    trap = TRAP[idx]
    rng = random.Random(SEED_BASE + tid)
    donors = gen_donors(rng, N, trap)
    sumg0 = counterfactual_total(donors)
    bfrac = 0.28 if trap else 0.50
    B = bfrac * sumg0

    out = []
    out.append("%d %d %.6f %.6f" % (N, K_MAX, R_MAX, B))
    for a, w in donors:
        out.append("%.6f %.6f" % (a, w))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
