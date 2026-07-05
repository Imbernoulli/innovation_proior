# TIER: strong
# Best Korobov rank-1 lattice: for each generator g the quadrats are
#   p_i = ( (i+0.5)/n , ((i*g mod n)+0.5)/n , ((i*g^2 mod n)+0.5)/n ),  i=0..n-1
# We sweep g over 1..n-1 and keep the generator whose EXACT 3D star discrepancy
# is smallest. This small search adapts to n and reliably beats any single fixed
# construction (the diagonal baseline or the fixed Halton set).
import sys

def star_discrepancy_3d(pts, n):
    xs = sorted(set([p[0] for p in pts] + [1.0]))
    ys = sorted(set([p[1] for p in pts] + [1.0]))
    zs = sorted(set([p[2] for p in pts] + [1.0]))
    best = 0.0
    for qx in xs:
        for qy in ys:
            for qz in zs:
                V = qx * qy * qz
                nc = 0
                no = 0
                for (x, y, z) in pts:
                    if x <= qx and y <= qy and z <= qz:
                        nc += 1
                        if x < qx and y < qy and z < qz:
                            no += 1
                dplus = nc / n - V
                dminus = V - no / n
                m = dplus if dplus > dminus else dminus
                if m > best:
                    best = m
    return best

def korobov(n, g):
    pts = []
    for i in range(n):
        a = (i + 0.5) / n
        b = (((i * g) % n) + 0.5) / n
        c = (((i * g * g) % n) + 0.5) / n
        pts.append((a, b, c))
    return pts

def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    best_g = 1
    best_d = None
    for g in range(1, n):
        d = star_discrepancy_3d(korobov(n, g), n)
        if best_d is None or d < best_d:
            best_d = d
            best_g = g
    pts = korobov(n, best_g)
    out = ["%.10f %.10f %.10f" % (x, y, z) for (x, y, z) in pts]
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
