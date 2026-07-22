# TIER: invalid
"""Plausible-looking but infeasible: drops the first slot length and injects
an absurd sentinel length instead, breaking the required multiset equality
(the ROM only has the slots it has -- you can't invent a new one)."""
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    idx += n  # skip f[]
    L = [int(data[idx + i]) for i in range(n)]; idx += n

    out = list(L[1:]) + [999999]
    print(" ".join(str(x) for x in out))


if __name__ == "__main__":
    main()
