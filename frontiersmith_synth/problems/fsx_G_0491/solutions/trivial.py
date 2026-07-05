# TIER: trivial
# Loose axis-aligned cubic lattice with spacing 1.6*diameter -- the reference baseline.
import sys, math

def main():
    Lx, Ly, Lz, r = map(float, sys.stdin.read().split()[:4])
    d = 2.0 * r
    sp = 1.6 * d
    def coords(L):
        if L < 2.0 * r:
            return []
        n = int(math.floor((L - 2.0 * r) / sp + 1e-9)) + 1
        return [r + k * sp for k in range(n)]
    xs, ys, zs = coords(Lx), coords(Ly), coords(Lz)
    out = []
    for x in xs:
        for y in ys:
            for z in zs:
                out.append("%.9f %.9f %.9f" % (x, y, z))
    sys.stdout.write("%d\n" % len(out))
    if out:
        sys.stdout.write("\n".join(out) + "\n")

main()
