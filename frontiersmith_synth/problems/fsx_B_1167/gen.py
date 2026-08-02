#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE instance of fsx_B_1167 (gravity-density-inversion) to stdout.

Physical model (mirrored byte-for-byte in verify.py): a subsurface cross-section is
discretized into an nx (columns) x nz (depth layers, z=1 shallowest .. nz deepest) grid.
Each cell (c, z) is treated, for computing its surface signature, as a point mass at its
cell center; the vertical-gravity point-mass kernel

    K(dc, z) = KGAIN * depth(z) / (dc^2 * CELL_W^2 + depth(z)^2)^1.5,   depth(z) = (z-0.5)*LAYER_H

gives cell (c,z)'s contribution to the surface reading at the station directly above
column cs, where dc = cs - c (a column-offset, so dc=0 means the station sits right above
the cell). K decays as roughly 1/depth(z)^2 at dc=0 -- shallow cells are FAR more
"sensitive" than deep ones, which is the whole depth ambiguity: a small mass placed
shallow and a big mass placed deep can produce comparable near-field readings.

A HIDDEN true body -- ONE contiguous horizontal span of columns [c_lo,c_hi] at ONE
depth layer z_true, uniform density rho_true (the layer-continuity prior: real buried
layers are laterally continuous, not scattered single cells) -- generates the surface
gravity field. Stations sit above every column; readings at EVEN columns are printed
here (the "given"/training transect); ODD columns are held out and re-derived only
inside verify.py from the SAME testId-seeded procedure (never printed here) so a
submission is graded on genuine reconstruction, not on echoing printed numbers.

Everything is seeded by testId only -> fully deterministic. testId ladder mixes SHALLOW
true bodies (z_true in {1,2}, where a naive shallow-only guess is basically correct) with
DEEP true bodies (z_true >= 4, six of the ten cases) where a shallow-only reconstruction
is structurally wrong (zero cell overlap with the truth) even if it happens to explain the
near-field numbers reasonably -- the trap.
"""
import sys, random

CELL_W = 2.0
LAYER_H = 2.0
KGAIN = 100.0
RHO_MAX = 10.0
NOISE_HALF = 0.02          # +-2% multiplicative sensor noise

# testId -> (nx, nz, z_true, w_true, rho_frac)
TABLE = {
    1:  (8,  4, 1, 2, 0.60),
    2:  (8,  4, 4, 2, 0.80),
    3:  (10, 5, 2, 3, 0.65),
    4:  (10, 5, 5, 3, 0.85),
    5:  (12, 5, 1, 3, 0.55),
    6:  (12, 6, 5, 3, 0.80),
    7:  (14, 6, 6, 4, 0.85),
    8:  (14, 6, 2, 4, 0.60),
    9:  (16, 7, 6, 4, 0.80),
    10: (16, 7, 7, 3, 0.90),
}


def depth(z):
    return (z - 0.5) * LAYER_H


def kernel(dc, z):
    dx = dc * CELL_W
    dpt = depth(z)
    return KGAIN * dpt / (dx * dx + dpt * dpt) ** 1.5


def true_body(test_id):
    """Return (nx, nz, z_true, c_lo, c_hi, rho_true, mass_max). Identical logic lives in
    verify.py so the checker can regenerate the hidden truth from testId alone."""
    nx, nz, z_true, w_true, rho_frac = TABLE[test_id]
    rho_true = rho_frac * RHO_MAX
    rng = random.Random(9001 + test_id * 131)
    c_lo = rng.randint(0, nx - w_true)
    c_hi = c_lo + w_true - 1
    true_mass = rho_true * w_true
    mass_max = round(1.8 * true_mass, 3)
    return nx, nz, z_true, c_lo, c_hi, rho_true, mass_max


def true_field(cs, z_true, c_lo, c_hi, rho_true):
    return rho_true * sum(kernel(cs - c, z_true) for c in range(c_lo, c_hi + 1))


def noise_factor(test_id, c):
    r = random.Random(5000 + test_id * 97 + c * 13)
    return 1.0 + r.uniform(-NOISE_HALF, NOISE_HALF)


def main():
    test_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if test_id not in TABLE:
        test_id = ((test_id - 1) % 10) + 1

    nx, nz, z_true, c_lo, c_hi, rho_true, mass_max = true_body(test_id)

    given_cols = [c for c in range(nx) if c % 2 == 0]
    lines = []
    lines.append("%d %d %d" % (test_id, nx, nz))
    lines.append("%.6f %.6f" % (RHO_MAX, mass_max))
    lines.append("%d" % len(given_cols))
    for c in given_cols:
        r = true_field(c, z_true, c_lo, c_hi, rho_true) * noise_factor(test_id, c)
        lines.append("%d %.8g" % (c, r))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
