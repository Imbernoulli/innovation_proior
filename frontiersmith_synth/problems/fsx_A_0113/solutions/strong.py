# TIER: strong
"""Weight-ordered largest-empty-disk placement + radius coordinate-ascent.

Process substations from heaviest to lightest load. For each one, scan a fine
grid of candidate centres and pick the centre whose largest feasible coverage
radius (limited by the region boundary and by all already-placed disks) is
greatest, then place the disk there at that radius. Concentrating the biggest
available disks on the heaviest loads is far stronger than a weight-blind grid.

A final set of coordinate-ascent sweeps (descending weight order) regrows each
disk to reclaim any slack opened up by later placements -- giving per-instance
gains on top of the greedy construction."""
import sys, math


def feasible_r(x, y, placed):
    r = min(x, 1.0 - x, y, 1.0 - y)
    for (px, py, pr) in placed:
        d = math.hypot(x - px, y - py) - pr
        if d < r:
            r = d
    return r


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    weights = [float(toks[1 + i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: -weights[i])

    G = 61                      # candidate-centre grid resolution
    coords = [t / (G - 1) for t in range(G)]

    placed = []
    disks = [None] * n
    for idx in order:
        best_r = -1.0
        best_xy = (0.5, 0.5)
        for gx in coords:
            for gy in coords:
                r = feasible_r(gx, gy, placed)
                if r > best_r:
                    best_r = r
                    best_xy = (gx, gy)
        if best_r < 0.0:
            best_r = 0.0
        disks[idx] = [best_xy[0], best_xy[1], best_r]
        placed.append((best_xy[0], best_xy[1], best_r))

    # radius coordinate-ascent sweeps
    for _ in range(6):
        for idx in order:
            xi, yi, _ = disks[idx]
            rmax = min(xi, 1.0 - xi, yi, 1.0 - yi)
            for j in range(n):
                if j == idx:
                    continue
                xj, yj, rj = disks[j]
                d = math.hypot(xi - xj, yi - yj) - rj
                if d < rmax:
                    rmax = d
            disks[idx][2] = max(0.0, rmax)

    lines = ["%.9f %.9f %.9f" % (disks[i][0], disks[i][1], disks[i][2]) for i in range(n)]
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
