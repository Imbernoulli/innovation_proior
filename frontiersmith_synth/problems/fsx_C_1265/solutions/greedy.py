# TIER: greedy
"""
The obvious recipe an average strong coder reaches for first, from two textbook instincts:

1. "The topmost die is farthest from the heat sink, so it is the hardest to cool -- size the
   via placement against ITS OWN power map." This looks past every lower die entirely: it
   never notices that several lower dies, individually unremarkable, can stack (through the
   shared vertical via path) into a hotter column than the top die's own single worst spot.

2. "It's a budgeted selection problem, so rank by value-per-cost and fill the knapsack
   greedily." This is the standard fractional-knapsack heuristic -- and it is the wrong
   objective here: the score is driven by the single WORST remaining column (a min-max), not
   by a sum, so chasing high value-density can spend the whole budget on cheap, moderately hot
   columns while leaving the one truly dominant hotspot (which may be expensive to via) fully
   untouched.

Neither instinct ever computes the depth-weighted, cross-die stacked profile.
"""
import sys


def main():
    data = sys.stdin.read().split()
    p = iter(data)

    def nx():
        return next(p)

    M = int(nx())
    N = int(nx())
    A = int(nx())
    nx(); nx()  # R0, Rv unused by this recipe
    a = [int(nx()) for _ in range(N)]
    P = [[int(nx()) for _ in range(N)] for _ in range(M)]

    # "value" = only the topmost die's own power map (single-die view, no vertical coupling).
    top = P[M - 1]
    order = sorted(range(N), key=lambda c: (-(top[c] / a[c]), c))

    x = [0] * N
    remaining = A
    for c in order:
        if a[c] <= remaining:
            x[c] = 1
            remaining -= a[c]

    sys.stdout.write(" ".join(map(str, x)) + "\n")


if __name__ == "__main__":
    main()
