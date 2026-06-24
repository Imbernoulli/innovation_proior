import sys
from itertools import combinations
from fractions import Fraction

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    pts = []
    for _ in range(n):
        xi = int(data[idx]); yi = int(data[idx + 1]); idx += 2
        pts.append((xi, yi))

    # Independent brute force: for every triple, compute twice the area via the
    # shoelace formula using Python's arbitrary-precision integers (no overflow
    # ever). Twice the area is always an integer. Output the maximum, 0 if no
    # triple exists.
    best = 0
    for (ax, ay), (bx, by), (cx, cy) in combinations(pts, 3):
        # shoelace: 2*signed area = ax*(by-cy)+bx*(cy-ay)+cx*(ay-by)
        twice_signed = ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)
        twice = abs(twice_signed)
        if twice > best:
            best = twice

    print(best)

if __name__ == "__main__":
    main()
