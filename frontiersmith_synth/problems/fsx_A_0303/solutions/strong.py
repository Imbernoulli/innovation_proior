# TIER: strong
# Ensemble of two constructions; emit whichever yields the larger total radius:
#   (A) single equal-radius ring (optimal for very small N), and
#   (B) an equal-radius hexagonal-lattice packing whose radius is pushed as large
#       as possible while still fitting N vials inside the carrier (dominates for
#       larger N, where filling the interior beats a lone ring).
import sys
import math

toks = sys.stdin.read().split()
N = int(toks[0])
R = float(toks[1])


def ring():
    s = math.sin(math.pi / N)
    r = R * s / (1.0 + s)
    d = R - r
    pts = []
    for k in range(N):
        th = 2.0 * math.pi * k / N
        pts.append((d * math.cos(th), d * math.sin(th), r))
    return pts


def hex_centers(r):
    if r <= 0:
        return []
    s = 2.0 * r
    dy = s * math.sqrt(3.0) / 2.0
    lim = R - r
    if lim < 0:
        return []
    J = int(lim / dy) + 2
    I = int((lim + s) / s) + 2
    C = []
    for j in range(-J, J + 1):
        y = j * dy
        xoff = (j & 1) * (s / 2.0)
        for i in range(-I, I + 1):
            x = i * s + xoff
            if x * x + y * y <= lim * lim + 1e-12:
                C.append((x, y))
    return C


def hexpack():
    lo, hi, best = 1e-4, R, 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if len(hex_centers(mid)) >= N:
            lo = mid
            best = mid
        else:
            hi = mid
    if best <= 0.0:
        return []
    C = hex_centers(best)
    C.sort(key=lambda c: c[0] * c[0] + c[1] * c[1])
    return [(x, y, best) for (x, y) in C[:N]]


def total(pts):
    return sum(p[2] for p in pts)


cands = [ring(), hexpack()]
best = max(cands, key=total)

out = [str(len(best))]
for (x, y, r) in best:
    out.append("%.10f %.10f %.10f" % (x, y, r))
sys.stdout.write("\n".join(out) + "\n")
