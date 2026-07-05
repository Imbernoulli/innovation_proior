# TIER: strong
"""Hexagonal (triangular) lattice packing. Denser than a square grid, so for the
same dome count N the domes can be larger => strictly higher sum of radii. Picks
the largest lattice spacing d that still yields >= N valid dome sites (inside the
pad, clear of the antenna), then sets each radius to ~half the spacing."""
import sys
import math

CX, CY = 0.5, 0.5


def sites(d, r0):
    """Hexagonal lattice dome centers with spacing d and radius r=0.49*d that fit
    inside the pad and clear the keep-out disk."""
    r = 0.49 * d
    dy = d * math.sqrt(3.0) / 2.0
    pts = []
    lo, hi = r, 1.0 - r
    if hi <= lo:
        return pts
    row = 0
    y = lo
    while y <= hi + 1e-12:
        xoff = lo + (d / 2.0 if (row % 2) else 0.0)
        x = xoff
        while x <= hi + 1e-12:
            if math.hypot(x - CX, y - CY) >= r0 + r + 1e-9:
                pts.append((x, y))
            x += d
        row += 1
        y = lo + row * dy
    return pts


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    r0 = float(toks[1])

    best = None
    d = 0.5
    while d >= 0.02:
        pts = sites(d, r0)
        if len(pts) >= N:
            best = (d, pts)
            break
        d -= 0.001

    if best is None:
        d = 0.02
        best = (d, sites(d, r0))

    d, pts = best
    r = 0.49 * d
    for (x, y) in pts[:N]:
        print("%.9f %.9f %.9f" % (x, y, r))


if __name__ == "__main__":
    main()
