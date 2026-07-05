# TIER: invalid
"""Infeasible: emits out-of-range entries (2), so strict validation rejects it -> Ratio 0."""
import sys


def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    out = [" ".join("2" for _ in range(N)) for _ in range(N)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
