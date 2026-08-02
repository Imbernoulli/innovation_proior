# TIER: greedy
"""Welfare-maximizing assignment, ignoring fairness entirely: give each item
WHOLLY to whichever agent values it most (ties -> lowest index). This is the
natural first instinct ("just maximize total value") and it literally achieves
the unconstrained welfare ceiling sum_j max_i v[i][j] -- but it has no mechanism
to avoid envy, so whenever multiple agents prize the same item it hands that
whole item to a single winner and everyone else is left envying the winner."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    v = [[int(next(it)) for _ in range(m)] for _ in range(n)]

    out_lines = []
    for j in range(m):
        best_i, best_v = 0, v[0][j]
        for i in range(1, n):
            if v[i][j] > best_v:
                best_v = v[i][j]
                best_i = i
        row = [0.0] * n
        row[best_i] = 1.0
        out_lines.append(" ".join(f"{x:.9f}" for x in row))
    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
