# TIER: strong
# Face-centered-cubic (close) packing -- the density-optimal lattice.
# FCC = integer points (a,b,c) with a+b+c even, scaled by sqrt(2) so the
# nearest-neighbour distance equals one diameter. We slide the lattice origin
# over a small deterministic grid of offsets and keep the placement that seats
# the most spheres in this particular bin (boundary-adapted).
import sys, math

S = math.sqrt(2.0)   # lattice step; nearest-neighbour dist = S*sqrt(2) = 2 = diameter

def pack(Lx, Ly, Lz, r, ox, oy, oz):
    lo = r
    hix, hiy, hiz = Lx - r, Ly - r, Lz - r
    x0, y0, z0 = lo + ox, lo + oy, lo + oz
    # index ranges so coordinates stay in [lo, hi]
    def rng(x0, hi):
        amin = int(math.floor((lo - x0) / S)) - 1
        amax = int(math.ceil((hi - x0) / S)) + 1
        return amin, amax
    axr = rng(x0, hix); ayr = rng(y0, hiy); azr = rng(z0, hiz)
    pts = []
    eps = 1e-9
    for a in range(axr[0], axr[1] + 1):
        x = x0 + S * a
        if x < lo - eps or x > hix + eps:
            continue
        for b in range(ayr[0], ayr[1] + 1):
            y = y0 + S * b
            if y < lo - eps or y > hiy + eps:
                continue
            # c must have parity making (a+b+c) even; pick correct start
            cstart = azr[0]
            if (a + b + cstart) % 2 != 0:
                cstart += 1
            for c in range(cstart, azr[1] + 1, 2):
                z = z0 + S * c
                if z < lo - eps or z > hiz + eps:
                    continue
                pts.append((x, y, z))
    return pts

def main():
    Lx, Ly, Lz, r = map(float, sys.stdin.read().split()[:4])
    best = None
    # deterministic offset sweep within one lattice cell
    grid = [0.0, S * 0.5, S * 0.25, S * 0.75]
    for ox in grid:
        for oy in grid:
            for oz in (0.0, S * 0.5):
                p = pack(Lx, Ly, Lz, r, ox, oy, oz)
                if best is None or len(p) > len(best):
                    best = p
    if best is None:
        best = []
    sys.stdout.write("%d\n" % len(best))
    if best:
        sys.stdout.write("\n".join("%.9f %.9f %.9f" % p for p in best) + "\n")

main()
