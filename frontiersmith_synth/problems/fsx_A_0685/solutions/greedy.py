# TIER: greedy
# "Grow the tile stone by stone": scan the plaza left to right and grab the first
# k quarry-approved cells you find, then translate by the natural uniform spacing
# 0,k,2k,...  This is the obvious recipe -- it never looks at *which* residue class
# (mod k) each stone belongs to, so whenever the quarry-approved stones happen to be
# unevenly spread across classes it silently produces a badly defective paving.
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    k = int(data[idx]); idx += 1
    allowed = [int(x) for x in data[idx:idx + n]]; idx += n
    # cost not needed by this tier

    M = n // k
    B = [i for i in range(n) if allowed[i]][:k]
    T = [t * k for t in range(M)]  # uniform spacing -- the "obvious" guess given k|n

    print(" ".join(map(str, B)))
    print(" ".join(map(str, T)))


if __name__ == "__main__":
    main()
