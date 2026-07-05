# TIER: greedy
# Random spread of n distinct weights within a TIGHT sub-interval [0, 0.7*M], deterministically
# seeded. Spreading the weights out gains a lot of difference-diversity over the AP baseline,
# but the tight interval forces many sum-collisions, so it stays well below a true Sidon packing.
import random
import sys


def main():
    toks = sys.stdin.read().split()
    n, M, k = int(toks[0]), int(toks[1]), int(toks[2])
    forb = set(int(x) for x in toks[3:3 + k])

    hi = max(n, int(M * 0.7))
    rnd = random.Random(1234 + 7 * n + M)
    A = set()
    guard = 0
    while len(A) < n and guard < 50 * n + 100000:
        v = rnd.randint(0, hi)
        if v not in forb and v not in A:
            A.add(v)
        guard += 1
    # fallback: if the tight interval somehow can't supply n distinct, fill upward
    c = 0
    while len(A) < n and c <= M:
        if c not in forb and c not in A:
            A.add(c)
        c += 1
    sys.stdout.write(" ".join(map(str, sorted(A))) + "\n")


if __name__ == "__main__":
    main()
