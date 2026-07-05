import sys, math

TOL = 1e-6
NMAX = 2_000_000            # hard cap on number of declared spheres (bounded read)

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def read_instance(path):
    tok = open(path).read().split()
    if len(tok) < 4:
        fail("bad instance")
    Lx, Ly, Lz, r = (float(tok[0]), float(tok[1]), float(tok[2]), float(tok[3]))
    return Lx, Ly, Lz, r

def cubic_baseline(L, r, sp):
    """Count of a loose axis-aligned cubic lattice (spacing sp) that fits in the bin.
    Centers start at r and step by sp; a sphere fits iff its center in [r, L-r]."""
    def n(x):
        if x < 2.0 * r - TOL:
            return 0
        return int(math.floor((x - 2.0 * r) / sp + 1e-9)) + 1
    return n(L[0]) * n(L[1]) * n(L[2])

def main():
    Lx, Ly, Lz, r = read_instance(sys.argv[1])
    L = (Lx, Ly, Lz)
    d = 2.0 * r
    d2min = (d - TOL) ** 2

    # ---------- internal baseline B (loose cubic lattice, spacing 1.6*diameter) ----------
    B = cubic_baseline(L, r, 1.6 * d)
    B = max(1, B)

    # ---------- parse participant artifact ----------
    tok = open(sys.argv[2]).read().split()
    if not tok:
        fail("empty output")
    try:
        N = int(tok[0])
    except Exception:
        fail("bad count")
    if N < 0:
        fail("negative count")
    if N > NMAX:
        fail("too many spheres")
    if len(tok) != 1 + 3 * N:
        fail("token count mismatch (need %d got %d)" % (1 + 3 * N, len(tok)))

    pts = []
    idx = 1
    for _ in range(N):
        try:
            x = float(tok[idx]); y = float(tok[idx + 1]); z = float(tok[idx + 2])
        except Exception:
            fail("non-numeric coordinate")
        idx += 3
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            fail("non-finite coordinate")
        # containment: sphere fully inside bin  <=>  center in [r, L-r]
        if x < r - TOL or x > Lx - r + TOL:
            fail("out of bin (x)")
        if y < r - TOL or y > Ly - r + TOL:
            fail("out of bin (y)")
        if z < r - TOL or z > Lz - r + TOL:
            fail("out of bin (z)")
        pts.append((x, y, z))

    # ---------- non-overlap check via uniform spatial hash (cell = diameter) ----------
    cell = d
    grid = {}
    for (x, y, z) in pts:
        cx = int(math.floor(x / cell)); cy = int(math.floor(y / cell)); cz = int(math.floor(z / cell))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for (ox, oy, oz) in grid.get((cx + dx, cy + dy, cz + dz), ()):
                        ddx = x - ox; ddy = y - oy; ddz = z - oz
                        if ddx * ddx + ddy * ddy + ddz * ddz < d2min:
                            fail("overlap")
        grid.setdefault((cx, cy, cz), []).append((x, y, z))

    # ---------- objective + normalized score (maximize count) ----------
    F = N
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
