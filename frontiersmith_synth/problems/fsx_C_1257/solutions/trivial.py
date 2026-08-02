# TIER: trivial
"""
Spend only about HALF of the energy budget on one fixed, precursor-blind sampling rate spread
across the whole horizon, and never touch the cheap precursor channel at all (Pc=0) --
deliberately wastes the other half of the battery. Distinctly weaker than `greedy` (which
spends the whole budget) and never in the same league as `strong` (which also exploits the
precursor channel).
"""
import sys


def main():
    data = sys.stdin.read().split()
    T = int(data[0])
    E = int(data[1])
    e_full = int(data[2])

    n_half = max(1, E // (2 * e_full))
    P0 = max(1, T // n_half)
    print(f"{P0} 0 {P0} 0")


if __name__ == "__main__":
    main()
