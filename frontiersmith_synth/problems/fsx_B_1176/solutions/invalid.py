# TIER: invalid
"""Deliberately infeasible: every appliance is claimed ON for the entire
trace. T (24..50) exceeds every archetype's maxOn dwell bound (4..16), so
the single run covering the whole sequence blows the max-dwell cap for
every appliance. Must score 0."""
import sys


def main():
    data = sys.stdin.read().split()
    T = int(data[0]); A = int(data[1])

    out = [str(A)]
    for _ in range(A):
        out.append(" ".join(["1"] * T))
    print("\n".join(out))


if __name__ == "__main__":
    main()
