# TIER: trivial
"""Reproduces the checker's own reference baseline exactly: enable every bond
option simultaneously at one single, fixed, late time step. No thought about
which bonds are target vs. decoy, no thought about strength at all."""
import sys

BASELINE_FRAC = 0.94


def main():
    data = sys.stdin.read().split()
    idx = 0
    N = int(data[idx]); idx += 1
    M = int(data[idx]); idx += 1
    Tmax = int(data[idx]); idx += 1
    theta0 = int(data[idx]); idx += 1
    # bond lines are irrelevant to this construction; skip them
    idx += 4 * M

    t = max(1, min(Tmax, int(round(Tmax * BASELINE_FRAC))))
    print(" ".join([str(t)] * M))


if __name__ == "__main__":
    main()
