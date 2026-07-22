# TIER: invalid
"""Infeasible on purpose: commits an absurd amount (far beyond any possible
free capital) into instrument 0 at period 1, and leaves every other period
blank-zero. The checker must reject this with Ratio: 0.0."""
import sys


def main():
    data = sys.stdin.read().split()
    T = int(data[0]); K = int(data[1])
    out_rows = []
    for t in range(1, T + 1):
        row = [0.0] * K
        if t == 1:
            row[0] = 1e18
        out_rows.append(" ".join(("%.6f" % x) for x in row))
    sys.stdout.write("\n".join(out_rows) + "\n")


if __name__ == "__main__":
    main()
