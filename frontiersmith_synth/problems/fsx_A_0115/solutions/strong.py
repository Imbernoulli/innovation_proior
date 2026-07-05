# TIER: strong
# Multi-restart, cooling-step local search on integer intensities to flatten the
# autocorrelation peak, driving the overlap constant c toward ~1.5-1.65.
# Fully deterministic (seed = 1000*n + restart).
import sys
import numpy as np

def c1(f):
    n = len(f)
    S = int(f.sum())
    M = int(np.convolve(f, f).max())
    return 2 * n * M / (S * S)

def solve(n, U):
    best = None
    bb = float("inf")
    for r in range(6):
        rng = np.random.default_rng(1000 * n + r)
        cur = np.clip(np.full(n, U // 2) + rng.integers(-150, 151, n), 1, U)
        bc = c1(cur)
        step = 150
        for it in range(9000):
            if it % 1800 == 1799:
                step = max(6, step // 2)
            i = int(rng.integers(n))
            d = int(rng.integers(-step, step + 1))
            old = cur[i]
            cur[i] = min(U, max(0, old + d))
            if cur.sum() > 0:
                c = c1(cur)
                if c <= bc:
                    bc = c
                    continue
            cur[i] = old
        if bc < bb:
            bb = bc
            best = cur.copy()
    return best

def main():
    tok = sys.stdin.read().split()
    n = int(tok[0]); U = int(tok[1])
    f = solve(n, U)
    print(" ".join(str(int(x)) for x in f))

if __name__ == "__main__":
    main()
