# TIER: invalid
"""Infeasible output: tune EVERY tower to channel 0. This violates the row/column/
district no-repeat rules everywhere (and overwrites givens), so the checker must
reject it -> Ratio 0.0."""
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    out = []
    for i in range(n):
        out.append(" ".join("0" for _ in range(n)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
