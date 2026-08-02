# TIER: greedy
"""The obvious textbook recipe: more ADC bits = better SNR per sample
(quantization noise falls exponentially in B), so maximize resolution
first. Spend a small, fixed, "reasonable-looking" filter order (K=1, don't
overthink it), no gain trick (G=0), and whatever budget is left over goes
to sample rate. This never checks whether the resulting sample rate still
gives the antialias filter (whose cutoff is pinned to R/2) enough headroom
to cover the target band -- on wideband instances with a tight budget, R
collapses far enough that the filter starts attenuating the band itself
while still letting interferer energy alias in."""
import sys

BMAX = 16
KDEF = 1


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    NBINS = int(next(it)); BUDGET = int(next(it)); WFILT = int(next(it))
    PFS = int(next(it)); NFLOOR = int(next(it))
    FLO = int(next(it)); FHI = int(next(it))
    for _ in range(NBINS):
        next(it); next(it)

    B = BMAX
    K = KDEF
    R = (BUDGET - WFILT * K) // B
    if R < 1:
        K = 0
        R = (BUDGET - WFILT * K) // B
    if R < 1:
        R = 1
    G = 0
    print(B, R, K, G)


if __name__ == "__main__":
    main()
