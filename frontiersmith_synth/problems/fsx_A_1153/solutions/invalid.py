# TIER: invalid
"""Dumps every tree into plot 1 -- violates the per-plot size requirement
for every other plot, so the checker must reject it with Ratio: 0.0."""
import sys


def main():
    data = sys.stdin.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = data[pos]
        pos += 1
        return v

    p = int(nxt())
    k = int(nxt())
    _sizes = [int(nxt()) for _ in range(k)]
    n = p - 1
    sys.stdout.write(" ".join(["1"] * n) + "\n")


if __name__ == "__main__":
    main()
