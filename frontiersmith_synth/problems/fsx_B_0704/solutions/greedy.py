# TIER: greedy
"""Textbook recipe: heat matters, so estimate ONE safe-ish temperature from the
billet-wide AVERAGE threshold (a single global summary statistic, ignoring how
that threshold is spatially distributed across grains) and hold it for as many
steps as the fuel budget affords, using the whole budget with no early stop.
This is the "obvious" plan a solver writes without probing the kinetics for a
possible fragile minority hiding inside the average.
"""
import sys


def main():
    data = sys.stdin.read().split()
    p = 0
    L = int(data[p]); p += 1
    Tmax = int(data[p]); p += 1
    C0 = int(data[p]); p += 1
    n_max = int(data[p]); p += 1
    B = int(data[p]); p += 1
    p += L  # skip d_i, unused by this recipe
    theta = [int(data[p + i]) for i in range(L)]; p += L

    mean_theta = sum(theta) / L
    T = int(mean_theta) - 1
    if T < 0:
        T = 0
    if T > Tmax:
        T = Tmax

    cost_per_step = C0 + T
    n = min(n_max, B // cost_per_step) if cost_per_step > 0 else n_max
    if n < 0:
        n = 0

    print(n)
    if n > 0:
        print(" ".join([str(T)] * n))


main()
