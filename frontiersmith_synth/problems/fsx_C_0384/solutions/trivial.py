# TIER: trivial
"""Build nothing new: echo the givens, leave every other cell empty.
Reproduces the checker baseline B -> Ratio ~ 0.1."""
import sys


def main():
    tok = sys.stdin.read().split()
    p = 0
    n = int(tok[p]); p += 1
    grid = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(tok[p]); p += 1
        grid.append(row)
    out = []
    for i in range(n):
        out.append(" ".join(grid[i][j] if grid[i][j] != "." else "." for j in range(n)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
