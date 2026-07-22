# TIER: invalid
"""Deliberately infeasible: stamps only the top-left quadrant of the pocket's bounding box
with size-1 tools, ignoring the rest -- most pocket cells stay uncovered, so the checker
must score this Ratio: 0.0."""
import sys


def main():
    data = sys.stdin.read().split("\n")
    H, W, K, C = map(int, data[0].split())
    grid = [data[2 + r] for r in range(H)]

    out = []
    n = 0
    for r in range(H // 3):
        row = grid[r]
        for c in range(W // 3):
            if row[c] == '#':
                out.append(f"1 {r} {c}")
                n += 1
    print(n)
    sys.stdout.write("\n".join(out))
    if out:
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
