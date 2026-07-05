# TIER: greedy
# Coarse single-cell hill-climb from the flat profile. Deterministic (seed = n).
# Beats the flat baseline but stops well short of the annealed strong construction.
import sys
import numpy as np

def c1(f):
    n = len(f)
    S = int(f.sum())
    M = int(np.convolve(f, f).max())
    return 2 * n * M / (S * S)

def main():
    tok = sys.stdin.read().split()
    n = int(tok[0]); U = int(tok[1])
    rng = np.random.default_rng(n)
    cur = np.full(n, U // 2, dtype=int)
    bc = c1(cur)
    for _ in range(400):
        i = int(rng.integers(n))
        d = int(rng.integers(-40, 41))
        old = cur[i]
        cur[i] = min(U, max(0, old + d))
        if cur.sum() > 0:
            c = c1(cur)
            if c < bc:
                bc = c
                continue
        cur[i] = old
    print(" ".join(str(int(x)) for x in cur))

if __name__ == "__main__":
    main()
