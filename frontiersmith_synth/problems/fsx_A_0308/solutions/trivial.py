# TIER: trivial
# Baseline: one multiplier primitive per non-zero harmonic (mode-3) fiber.
#   primitive = e_bus(i) (x) e_line(j) (x) fiber_over_k   -> R = # non-zero (i,j) fibers = B0.
# This reproduces the checker's internal baseline, so it scores ~0.1 by construction.
import sys


def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    B = int(next(it)); L = int(next(it)); H = int(next(it))
    T = [[[0] * H for _ in range(L)] for _ in range(B)]
    for k in range(H):
        for i in range(B):
            for j in range(L):
                T[i][j][k] = int(next(it))

    prims = []
    for i in range(B):
        for j in range(L):
            fib = [T[i][j][k] for k in range(H)]
            if any(v != 0 for v in fib):
                a = [1 if x == i else 0 for x in range(B)]
                b = [1 if x == j else 0 for x in range(L)]
                prims.append((a, b, fib))

    out = [str(len(prims))]
    for a, b, c in prims:
        out.append(" ".join(map(str, a)))
        out.append(" ".join(map(str, b)))
        out.append(" ".join(map(str, c)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
