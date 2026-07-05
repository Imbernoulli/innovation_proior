# TIER: greedy
# Tight axis-aligned cubic lattice: spacing = exactly one diameter.
# Much denser than the loose baseline, but wastes the ~26% interstitial space
# that a close-packed arrangement would fill.
import sys, math

def main():
    Lx, Ly, Lz, r = map(float, sys.stdin.read().split()[:4])
    d = 2.0 * r
    sp = d
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
