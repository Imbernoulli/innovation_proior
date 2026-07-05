# TIER: greedy
# Brute-force over SMALL (support <= 2) integer exponent vectors, exponents in
# [-3,3], picking the one that minimizes the scale-free log-spread on the TRAIN
# rows.  A two-channel ratio can only PARTIALLY cancel the physical variation
# (the real invariant needs three channels), so this lands well above trivial
# but well below the strong solver.
import sys, math, itertools

NCOLS = 6


def read_train():
    data = sys.stdin.read().split("\n")
    ncols, n = map(int, data[1].split())
    rows = []
    for line in data[3:3 + n]:
        if line.strip():
            rows.append([float(x) for x in line.split()])
    return rows


def spread(a, logcols, means, norm2):
    n = len(logcols[0])
    s1 = s2 = 0.0
    for j in range(n):
        p = 0.0
        for i in range(NCOLS):
            if a[i]:
                p += a[i] * (logcols[i][j] - means[i])
        s1 += p
        s2 += p * p
    var = s2 / n - (s1 / n) ** 2
    if var < 0:
        var = 0.0
    return math.sqrt(var / norm2)


def main():
    rows = read_train()
    logcols = [[math.log(r[i]) for r in rows] for i in range(NCOLS)]
    means = [sum(c) / len(c) for c in logcols]

    best = None
    best_s = float("inf")
    vals = [-3, -2, -1, 1, 2, 3]
    for i, k in itertools.combinations(range(NCOLS), 2):
        for ai in vals:
            for ak in vals:
                a = [0] * NCOLS
                a[i] = ai
                a[k] = ak
                norm2 = ai * ai + ak * ak
                s = spread(a, logcols, means, norm2)
                if s < best_s:
                    best_s = s
                    best = a
    print(" ".join(str(x) for x in best))


main()
