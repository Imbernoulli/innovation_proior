import sys

# Warehouse-bin sphere packing instance generator.
# `python3 gen.py <testId>` prints ONE instance to stdout.
# Instance line:  Lx Ly Lz r
#   - a rectangular storage bin of interior dimensions Lx x Ly x Lz
#   - r = radius of the identical spherical goods (unit balls, r = 1)
# testId 1..N = difficulty ladder: bins grow small -> large, with
# deliberately non-integer / non-commensurate aspect ratios so that no
# axis-aligned lattice tiles the bin perfectly (no cheap closed-form optimum).

def main():
    i = int(sys.argv[1])
    r = 1.0

    # monotonically growing base scale (small -> large)
    base = 9.0 + 2.35 * i                      # i=1 -> 11.35 ... i=10 -> 32.5

    # per-test irrational-ish aspect multipliers (avoid perfect tiling / commensurability)
    ax = 1.0
    ay = 0.81 + 0.031 * ((i * 7) % 5)          # ~0.81 .. 0.934
    az = 1.09 + 0.027 * ((i * 4) % 3)          # ~1.09 .. 1.144

    # small deterministic per-test perturbation so bin faces are never round numbers
    px = 0.137 * ((i * 3) % 7)
    py = 0.113 * ((i * 5) % 6)
    pz = 0.101 * ((i * 2) % 8)

    Lx = base * ax + px
    Ly = base * ay + py
    Lz = base * az + pz

    sys.stdout.write("%.6f %.6f %.6f %.6f\n" % (Lx, Ly, Lz, r))

if __name__ == "__main__":
    main()
