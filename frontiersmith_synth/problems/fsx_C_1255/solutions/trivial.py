# TIER: trivial
"""Reproduces the checker's own baseline construction exactly: fixed
resolution B=6, fixed filter order K=2, no gain (G=0), and whatever sample
rate the leftover budget buys. No allocation reasoning at all."""
import sys

B0, K0, G0 = 6, 2, 0


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    NBINS = int(next(it)); BUDGET = int(next(it)); WFILT = int(next(it))
    PFS = int(next(it)); NFLOOR = int(next(it))
    FLO = int(next(it)); FHI = int(next(it))
    for _ in range(NBINS):
        next(it); next(it)

    R0 = (BUDGET - WFILT * K0) // B0
    if R0 < 1:
        R0 = 1
    print(B0, R0, K0, G0)


if __name__ == "__main__":
    main()
