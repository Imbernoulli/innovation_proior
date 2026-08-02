# TIER: trivial
"""Explain-the-peak: find the single given station with the largest reading and place ONE
cell of density at the shallowest layer (z=1) directly under it, with magnitude chosen so
that cell alone reproduces that one peak reading exactly (clipped to bounds/budget). No
attempt to fit any other station, no attempt to reason about depth or lateral spread at
all. This is the checker's own internal reference construction (B), so this solution's
ratio is exactly 0.1 on every case by definition."""
import sys

CELL_W = 2.0
LAYER_H = 2.0
KGAIN = 100.0


def depth(z):
    return (z - 0.5) * LAYER_H


def kernel(dc, z):
    dx = dc * CELL_W
    dpt = depth(z)
    return KGAIN * dpt / (dx * dx + dpt * dpt) ** 1.5


def main():
    data = sys.stdin.read().split()
    idx = 0
    test_id = int(data[idx]); idx += 1
    nx = int(data[idx]); idx += 1
    nz = int(data[idx]); idx += 1
    rho_max = float(data[idx]); idx += 1
    mass_max = float(data[idx]); idx += 1
    ns_given = int(data[idx]); idx += 1
    given = []
    for _ in range(ns_given):
        c = int(data[idx]); idx += 1
        r = float(data[idx]); idx += 1
        given.append((c, r))

    grid = [[0.0] * nx for _ in range(nz + 1)]
    c_peak, r_peak = max(given, key=lambda x: x[1])
    mag = r_peak / kernel(0, 1)
    if mag < 0.0:
        mag = 0.0
    if mag > rho_max:
        mag = rho_max
    grid[1][c_peak] = mag

    total = mag
    if total > mass_max and total > 0:
        f = mass_max / total
        grid[1][c_peak] *= f

    out = []
    for z in range(1, nz + 1):
        out.append(" ".join("%.6f" % v for v in grid[z]))
    print("\n".join(out))


if __name__ == "__main__":
    main()
