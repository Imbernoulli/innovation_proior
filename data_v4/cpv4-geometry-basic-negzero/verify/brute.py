import sys
from fractions import Fraction

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    pts = []
    for _ in range(n):
        xx = int(data[idx]); yy = int(data[idx + 1]); idx += 2
        pts.append((xx, yy))

    if n < 2:
        print("NONE")
        return

    # Independent brute force: enumerate every ordered pair i<j and compute the
    # signed area*2 using exact rational arithmetic (Fraction) to be totally sure
    # there is no integer-overflow or sign mistake. Then take the maximum.
    best = None
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            cr = Fraction(xi) * Fraction(yj) - Fraction(xj) * Fraction(yi)
            if best is None or cr > best:
                best = cr
    # best is an integer-valued Fraction
    print(int(best))

main()
