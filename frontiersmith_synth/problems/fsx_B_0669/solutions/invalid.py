# TIER: invalid
"""Emits a non-finite value -- must be rejected (Ratio: 0.0)."""
import sys


def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    out = ["0.000000"] * N
    if N >= 1:
        out[N // 2] = "nan"
    print("\n".join(out))


if __name__ == "__main__":
    main()
