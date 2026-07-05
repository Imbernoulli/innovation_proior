# TIER: strong
# Seeded simulated annealing over station sets, minimizing R = |A+A|/|A-A|.
# Deterministic: all randomness comes from the instance seed; iteration budget is a
# fixed function of n (never wall-clock). Reaches R ~ 0.62-0.71 across the ladder.
import sys
import random
import math


def ratio(A):
    P = set()
    D = set()
    for a in A:
        for b in A:
            P.add(a + b)
            D.add(a - b)
    return len(P) / len(D)


def main():
    toks = sys.stdin.read().split()
    n, M, seed = int(toks[0]), int(toks[1]), int(toks[2])
    rng = random.Random(seed ^ 0x5DEECE66)
    iters = min(11000, 240000 // n)
    restarts = 2
    best_global = 9.0
    best_A = None
    for _ in range(restarts):
        A = set(rng.sample(range(M + 1), n))
        cur = ratio(A)
        best = cur
        bestA = set(A)
        for t in range(iters):
            T = 0.05 * (1 - t / iters) + 0.002
            out = rng.choice(tuple(A))
            ins = rng.randrange(M + 1)
            if ins in A:
                continue
            B = set(A)
            B.discard(out)
            B.add(ins)
            r = ratio(B)
            if r <= cur or rng.random() < math.exp((cur - r) / T):
                A = B
                cur = r
                if r < best:
                    best = r
                    bestA = set(A)
        if best < best_global:
            best_global = best
            best_A = bestA
    sys.stdout.write(" ".join(map(str, sorted(best_A))) + "\n")


if __name__ == "__main__":
    main()
