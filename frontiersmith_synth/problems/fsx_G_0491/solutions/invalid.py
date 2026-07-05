# TIER: invalid
# Emits overlapping spheres (spacing = radius, far below one diameter) -> infeasible.
import sys, math

def main():
    Lx, Ly, Lz, r = map(float, sys.stdin.read().split()[:4])
    sp = r  # way too tight -> guaranteed overlaps
    def coords(L):
        n = max(1, int((L - 2.0 * r) / sp) + 1)
        return [r + k * sp for k in range(n)]
    xs, ys, zs = coords(Lx), coords(Ly), coords(Lz)
    out = []
    for x in xs:
        for y in ys:
            for z in zs:
                out.append("%.9f %.9f %.9f" % (x, y, z))
    sys.stdout.write("%d\n" % len(out))
    sys.stdout.write("\n".join(out) + "\n")

main()
