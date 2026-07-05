# TIER: greedy
# Seeded random-restart search: sample many random k-subsets of [0,M] and keep the
# one with the largest sumset |A+A|. A generic-but-improvable heuristic that already
# crushes the arithmetic-progression baseline, but leaves headroom above it.
import sys
import random


def ss(A):
    s = set()
    for x in A:
        for y in A:
            s.add(x + y)
    return len(s)


def main():
    tok = sys.stdin.read().split()
    k, M = int(tok[0]), int(tok[1])
    rng = random.Random(20240624 + 131 * k + M)
    pool = list(range(M + 1))
    best = None
    bestv = -1
    restarts = 400
    for _ in range(restarts):
        A = rng.sample(pool, k)
        v = ss(A)
        if v > bestv:
            bestv = v
            best = A
    best = sorted(best)
    out = [str(len(best))] + [str(x) for x in best]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
