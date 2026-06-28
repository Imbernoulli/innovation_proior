#!/usr/bin/env python3
# Independent O(n^2) brute force oracle for the closest-pair problem.
# Reads stdin: n, then n lines/pairs "x y". Prints min squared distance,
# or -1 if n < 2.
import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    pts = []
    for _ in range(n):
        x = int(data[idx]); y = int(data[idx + 1]); idx += 2
        pts.append((x, y))
    if n < 2:
        print(-1)
        return
    best = None
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            dx = xi - pts[j][0]
            dy = yi - pts[j][1]
            d2 = dx * dx + dy * dy
            if best is None or d2 < best:
                best = d2
    print(best)

if __name__ == "__main__":
    main()
