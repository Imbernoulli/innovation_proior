import sys
from math import gcd

# Brute oracle: count lattice points STRICTLY inside a simple polygon by
# scanning every lattice point in the bounding box and testing each one
# with exact integer arithmetic. Slow (O(area + n) per point) but obviously
# correct for small coordinates.

def on_segment(px, py, ax, ay, bx, by):
    # is point P on segment A-B (inclusive of endpoints)?
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if cross != 0:
        return False
    # collinear: check bounding box
    if min(ax, bx) <= px <= max(ax, bx) and min(ay, by) <= py <= max(ay, by):
        return True
    return False

def point_in_polygon(px, py, xs, ys):
    n = len(xs)
    # boundary points are NOT strictly inside
    for i in range(n):
        j = (i + 1) % n
        if on_segment(px, py, xs[i], ys[i], xs[j], ys[j]):
            return False
    # ray casting to the +x direction using a half-open rule on y
    inside = False
    for i in range(n):
        j = (i + 1) % n
        ax, ay = xs[i], ys[i]
        bx, by = xs[j], ys[j]
        if (ay > py) != (by > py):
            # x coordinate of edge at scanline y = py, compared as a rational
            # px < ax + (bx-ax)*(py-ay)/(by-ay)
            # multiply out keeping sign of (by-ay)
            lhs = (px - ax) * (by - ay)
            rhs = (bx - ax) * (py - ay)
            if (by - ay) > 0:
                cond = lhs < rhs
            else:
                cond = lhs > rhs
            if cond:
                inside = not inside
    return inside

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    xs = []; ys = []
    for _ in range(n):
        xs.append(int(data[idx])); idx += 1
        ys.append(int(data[idx])); idx += 1
    minx = min(xs); maxx = max(xs)
    miny = min(ys); maxy = max(ys)
    count = 0
    for px in range(minx, maxx + 1):
        for py in range(miny, maxy + 1):
            if point_in_polygon(px, py, xs, ys):
                count += 1
    print(count)

if __name__ == "__main__":
    main()
