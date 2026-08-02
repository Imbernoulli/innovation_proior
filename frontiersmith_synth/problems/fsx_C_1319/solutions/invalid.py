# TIER: invalid
"""Emits a schedule that is short by one token (and, for good measure,
contains an out-of-range enable time) -- must be rejected with Ratio: 0.0."""
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    N = int(data[idx]); idx += 1
    M = int(data[idx]); idx += 1
    Tmax = int(data[idx]); idx += 1
    # Print only M-1 tokens (too few), and make the ones we do print garbage
    # (enable time 0, which is out of [1, Tmax] anyway).
    n_out = max(0, M - 1)
    print(" ".join(["0"] * n_out))


if __name__ == "__main__":
    main()
