# TIER: strong
# Best rank-1 lattice: sweep the generator g over 1..n-1 for the point set
#   p_i = ((i+0.5)/n, ((i*g mod n)+0.5)/n)
# and keep the generator whose EXACT star discrepancy is smallest. This small search
# reliably beats any single fixed construction (diagonal or Hammersley).
import sys

def star_discrepancy(pts, n):
    xs = sorted(set([p[0] for p in pts] + [1.0]))
    ys = sorted(set([p[1] for p in pts] + [1.0]))
    best = 0.0
    for qx in xs:
        for qy in ys:
            V = qx * qy
            nc = 0
            no = 0
            for (x, y) in pts:
                if x <= qx and y <= qy:
                    nc += 1
                    if x < qx and y < qy:
                        no += 1
            dplus = nc / n - V
            dminus = V - no / n
            m = dplus if dplus > dminus else dminus
            if m > best:
                best = m
    return best

def lattice(n, g):
    return [((i + 0.5) / n, (((i * g) % n) + 0.5) / n) for i in range(n)]

def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    best_g = 1
    best_d = None
    for g in range(1, n):
        d = star_discrepancy(lattice(n, g), n)
        if best_d is None or d < best_d:
            best_d = d
            best_g = g
    pts = lattice(n, best_g)
    out = ["%.10f %.10f" % (x, y) for (x, y) in pts]
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
