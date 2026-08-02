# TIER: greedy
"""Textbook 'CD bias' correction: dilate every target pixel by a FIXED
Chebyshev radius of 1 to fight the average shrinkage caused by the blur.
This is the obvious first fix an engineer reaches for -- one global bias,
with no awareness of local feature density. It helps isolated small
features but bridges tightly pitched features together, so it does not
track the true optical model or the dose-latitude requirement."""
import sys

R = 1


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    rows = data[1:1 + n]
    target = [[1 if c == "1" else 0 for c in row] for row in rows]

    mask = [[0] * n for _ in range(n)]
    for x in range(n):
        for y in range(n):
            if target[x][y]:
                for dx in range(-R, R + 1):
                    xx = x + dx
                    if xx < 0 or xx >= n:
                        continue
                    for dy in range(-R, R + 1):
                        yy = y + dy
                        if 0 <= yy < n:
                            mask[xx][yy] = 1

    out = ["".join(str(c) for c in row) for row in mask]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
