# TIER: greedy
"""The obvious first pass at "read density off surface gravity": invert each given
station INDEPENDENTLY and assume every source sits in the SHALLOWEST layer (z=1), with
no lateral coupling between columns at all -- rho[1][c] = reading(c) / K(dc=0, z=1) for
every given column c, everything else left at 0. This is exactly the textbook mistake the
family is built around: because the point-mass kernel decays as roughly 1/depth^2, the
shallowest layer is always the "cheapest" (least magnitude needed) explanation of ANY
near-field reading, so a solver that doesn't reason about depth at all will always land
here by default. It genuinely explains the numbers reasonably well locally (the fit
term looks fine) -- but every single cell it activates is at z=1, so whenever the true
body sits deeper (z_true >= 4, six of the ten cases) its recovered region has ZERO
overlap with the truth: an unweighted, depth-blind pass always plasters density into the
top row, no matter how deep the real source is."""
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
    k0 = kernel(0, 1)
    for c, r in given:
        v = r / k0
        if v < 0.0:
            v = 0.0
        if v > rho_max:
            v = rho_max
        grid[1][c] = v

    total = sum(grid[1][c] for c in range(nx))
    if total > mass_max and total > 0:
        f = mass_max / total
        for c in range(nx):
            grid[1][c] *= f

    out = []
    for z in range(1, nz + 1):
        out.append(" ".join("%.6f" % v for v in grid[z]))
    print("\n".join(out))


if __name__ == "__main__":
    main()
