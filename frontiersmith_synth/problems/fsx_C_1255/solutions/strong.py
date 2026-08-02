# TIER: strong
"""The insight: R is not really a free continuous knob -- once you fix how
much of the shared budget goes to resolution (B) and filter order (K), the
achievable sample rate R = floor((BUDGET - WFILT*K) / B) is DETERMINED.
That collapses the apparent 4-dimensional allocation problem to a small,
enumerable search over (B, K, G): B in [1,16] and K in [0,8] are only 144
combinations, and for each the resulting R (hence the antialias cutoff
fc=R/2) and the resulting FS=PFS/2^G (12 more choices) can be evaluated
exactly. This directly exploits the family's hook: instead of maximizing
resolution in isolation (greedy's mistake), it looks at where the actual
signal/interferer energy sits in THIS instance's spectrum and picks
whichever (B,K,G) triple -- possibly a modest resolution, generous rate,
and just enough filter order -- gives the best exact SNR. No single named
textbook algorithm is "the" answer; it's a joint reformulation + exact
search over the budget-coupled variables."""
import sys
from fractions import Fraction as Fr

BMAX = 16
KMAX = 8
GMAX = 12


def atten(f, R, K):
    if K == 0:
        return Fr(1)
    ratio = Fr(2 * f, R)
    return Fr(1) / (Fr(1) + ratio ** (2 * K))


def evaluate(bins, FLO, FHI, PFS, NFLOOR, B, R, K, G):
    FS = Fr(PFS, 2 ** G)
    quant = FS / Fr(4) ** B
    sig = Fr(0)
    ali = Fr(0)
    clip_total = Fr(0)
    for f, p in bins:
        a = atten(f, R, K)
        fp = Fr(p) * a
        c = fp - FS if fp > FS else Fr(0)
        usable = fp - c
        clip_total += c
        if FLO <= f <= FHI:
            sig += usable
        elif 2 * f > R:
            ali += usable
    noise = quant + ali + clip_total + Fr(NFLOOR)
    return sig, noise


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    NBINS = int(next(it)); BUDGET = int(next(it)); WFILT = int(next(it))
    PFS = int(next(it)); NFLOOR = int(next(it))
    FLO = int(next(it)); FHI = int(next(it))
    bins = []
    for _ in range(NBINS):
        f = int(next(it)); p = int(next(it))
        bins.append((f, p))

    best = None  # (SNR, B, R, K, G)
    for B in range(1, BMAX + 1):
        for K in range(0, KMAX + 1):
            rem = BUDGET - WFILT * K
            if rem < B:
                continue
            R = rem // B
            if R < 1:
                continue
            for G in range(0, GMAX + 1):
                sig, noise = evaluate(bins, FLO, FHI, PFS, NFLOOR, B, R, K, G)
                snr = sig / noise
                if best is None or snr > best[0]:
                    best = (snr, B, R, K, G)

    _, B, R, K, G = best
    print(B, R, K, G)


if __name__ == "__main__":
    main()
