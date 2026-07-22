# TIER: invalid
"""Deliberately infeasible: emits matrix values far outside the allowed
[-Jmax, Jmax] range (3x Jmax) and, for extra measure, an asymmetric entry
(J[0][1] != J[1][0]). Must score 0."""
import sys


def main():
    data = sys.stdin.read().split()
    T = int(data[1])
    Jmax = int(data[2])

    big = Jmax * 3
    J = [[big] * T for _ in range(T)]
    if T >= 2:
        J[0][1] = big + 1  # also breaks symmetry vs J[1][0] = big
    for row in J:
        print(" ".join(str(x) for x in row))


if __name__ == "__main__":
    main()
