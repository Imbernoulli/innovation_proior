# TIER: strong
# Search integer exponent vectors of support <= 3 (exponents in [-3,3]) and pick
# the one whose log-group is most invariant, scored by the WORST of two disjoint
# TRAIN sub-regimes (split by the bulk Reynolds proxy x0*x1/x3).  Requiring
# invariance across sub-regimes rejects directions that merely fit the TRAIN-only
# latent coupling and instead recovers the genuine three-channel physical
# invariant, which also generalizes to the held-out higher-Reynolds regime.
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


def spread_on(a, logsub, means, norm2):
    n = len(logsub[0])
    s1 = s2 = 0.0
    for j in range(n):
        p = 0.0
        for i in range(NCOLS):
            if a[i]:
                p += a[i] * (logsub[i][j] - means[i])
        s1 += p
        s2 += p * p
    var = s2 / n - (s1 / n) ** 2
    if var < 0:
        var = 0.0
    return math.sqrt(var / norm2)


def main():
    rows = read_train()
    # split into two Reynolds sub-regimes (proxy Re ~ U*L/nu = x0*x1/x3)
    keyed = sorted(rows, key=lambda r: r[0] * r[1] / r[3])
    half = len(keyed) // 2
    subs = [keyed[:half], keyed[half:]]
    logsubs = []
    means = []
    for sub in subs:
        lc = [[math.log(r[i]) for r in sub] for i in range(NCOLS)]
        logsubs.append(lc)
        means.append([sum(c) / len(c) for c in lc])

    vals = [-3, -2, -1, 1, 2, 3]
    best = None
    best_score = float("inf")
    for k in (2, 3):
        for cols in itertools.combinations(range(NCOLS), k):
            for combo in itertools.product(vals, repeat=k):
                a = [0] * NCOLS
                for idx, c in zip(cols, combo):
                    a[idx] = c
                norm2 = sum(x * x for x in a)
                worst = 0.0
                for m in range(2):
                    s = spread_on(a, logsubs[m], means[m], norm2)
                    if s > worst:
                        worst = s
                # mild parsimony tie-break matching the checker's penalty
                score = worst * (10 ** (0.02 * sum(abs(x) for x in a) / max(1, 1)))
                if score < best_score:
                    best_score = score
                    best = a
    print(" ".join(str(x) for x in best))


main()
