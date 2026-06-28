#!/usr/bin/env python3
import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    pts = []
    total = 0
    for _ in range(n):
        x = int(data[idx]); y = int(data[idx+1]); w = int(data[idx+2]); idx += 3
        pts.append((x, y, w))
        total += w
    if n == 0 or total == 0:
        print(0)
        return

    # Brute force: the optimum integer meeting point lies in the bounding box of
    # the sites (moving inside the box never increases L1). Scan EVERY integer
    # grid point in [minx,maxx] x [miny,maxy] and take the minimum total weighted
    # Manhattan distance. Obviously correct; only feasible for small coordinates.
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    best = None
    for gx in range(minx, maxx + 1):
        for gy in range(miny, maxy + 1):
            cost = 0
            for (x, y, w) in pts:
                cost += (abs(gx - x) + abs(gy - y)) * w
            if best is None or cost < best:
                best = cost
    print(best)

if __name__ == "__main__":
    main()
