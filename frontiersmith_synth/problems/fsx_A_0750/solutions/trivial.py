# TIER: trivial
"""All-size-1 stamps, row-major order. Always feasible, one tool used -> the checker's own
baseline construction (matches B exactly => Ratio ~ 0.1)."""
import sys


def main():
    data = sys.stdin.read().split("\n")
    H, W, K, C = map(int, data[0].split())
    grid = [data[2 + r] for r in range(H)]

    out = []
    n = 0
    for r in range(H):
        row = grid[r]
        for c in range(W):
            if row[c] == '#':
                out.append(f"1 {r} {c}")
                n += 1
    print(n)
    sys.stdout.write("\n".join(out))
    if out:
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
