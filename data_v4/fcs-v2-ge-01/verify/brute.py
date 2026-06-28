#!/usr/bin/env python3
# Brute-force oracle for "maximum-area triangle over n points".
# Output = max over all unordered triples of |cross product| = max(2 * area).
# O(n^3); obviously correct. Reads the same stdin format as sol.cpp.
import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    pts = []
    for _ in range(n):
        x = int(next(it)); y = int(next(it))
        pts.append((x, y))
    best = 0
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            ax, ay = xj - xi, yj - yi
            for k in range(j + 1, n):
                xk, yk = pts[k]
                bx, by = xk - xi, yk - yi
                area2 = abs(ax * by - ay * bx)
                if area2 > best:
                    best = area2
    print(best)

if __name__ == "__main__":
    main()
