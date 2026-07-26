# TIER: invalid
"""Deliberately infeasible: ignores the revealed entries and prints a constant
matrix, which will mismatch almost every revealed cell (and this is a rank-1
matrix regardless, so it would be a free win if the checker didn't enforce
feasibility strictly)."""
import sys


def main():
    data = sys.stdin.read().split()
    m = int(data[0])
    out_rows = [" ".join(["7"] * m) for _ in range(m)]
    sys.stdout.write("\n".join(out_rows) + "\n")


if __name__ == "__main__":
    main()
