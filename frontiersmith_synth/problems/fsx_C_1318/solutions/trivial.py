# TIER: trivial
"""Reproduce the checker's own baseline: try every element alone (all others
at zero), each filled to the largest amount that is itself feasible among
amounts confined to the first three bands (X <= 3*W-1); submit whichever
single element gives the best result."""
import math
import sys


def main():
    toks = sys.stdin.read().split()
    idx = 0
    K = int(toks[idx]); idx += 1
    W = int(toks[idx]); idx += 1
    numBins = int(toks[idx]); idx += 1
    s = [int(toks[idx + i]) for i in range(K)]; idx += K
    b = [int(toks[idx + i]) for i in range(K)]; idx += K
    T = [int(toks[idx + i]) for i in range(numBins)]; idx += numBins

    limit_bin = min(numBins - 1, 2)
    max_bx = (limit_bin + 1) * W - 1

    best_val = -1.0
    best_i = 0
    best_bx = 0
    for i in range(K):
        Bx_i = 0
        for cand in range(0, max_bx + 1):
            cbin = cand // W
            if b[i] * cand <= T[cbin]:
                Bx_i = cand
        val = s[i] * math.sqrt(Bx_i)
        if val > best_val:
            best_val = val
            best_i = i
            best_bx = Bx_i

    x = [0] * K
    x[best_i] = best_bx
    print(' '.join(map(str, x)))


if __name__ == "__main__":
    main()
