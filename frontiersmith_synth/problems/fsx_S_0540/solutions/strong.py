# TIER: strong
# Insight: model the INVARIANTS of the map, not the map.  The gentle logbook
# already pins the two conserved-quantity constants (the mass ratio r and the
# velocity ceiling c) even though the pointwise map looks almost linear there.
# We recover (r, c) by fitting the exact two-ball collision map -- the unique map
# consistent with the momentum-like and energy-like invariants and the physical
# (exchange) branch -- to the logged (u1,u2)->(v1,v2) pairs, then EMIT that exact
# closed form.  A conservation law is regime-independent, so the recovered map
# extrapolates to the violent ceiling regime where the black-box polynomial
# blows up.  The coarse recovery + irreducible held-out noise leave headroom < 1.
import sys
import math


def gamma(u, c):
    return 1.0 / math.sqrt(abs(1.0 - (u / c) ** 2))


def post(u1, u2, r, c):
    g1 = gamma(u1, c)
    g2 = gamma(u2, c)
    p = g1 * u1 + g2 * r * u2
    E = g1 + g2 * r
    V = p / E
    c2 = c * c
    w1 = (u1 - V) / (1.0 - u1 * V / c2)
    v1 = (V - w1) / (1.0 - w1 * V / c2)
    w2 = (u2 - V) / (1.0 - u2 * V / c2)
    v2 = (V - w2) / (1.0 - w2 * V / c2)
    return v1, v2


def resid(rows, r, c):
    umax = max(max(abs(u1), abs(u2)) for u1, u2, _, _ in rows)
    if c <= umax + 1e-6 or r <= 0.0:
        return 1e30
    s = 0.0
    for u1, u2, o1, o2 in rows:
        v1, v2 = post(u1, u2, r, c)
        s += (v1 - o1) ** 2 + (v2 - o2) ** 2
    return s


def golden_min(f, lo, hi, iters=40):
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    x1 = b - gr * (b - a)
    x2 = a + gr * (b - a)
    f1, f2 = f(x1), f(x2)
    for _ in range(iters):
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - gr * (b - a)
            f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + gr * (b - a)
            f2 = f(x2)
    return (a + b) / 2.0


def main():
    data = sys.stdin.read().split("\n")
    n = int(data[0].split()[0])
    rows = []
    for ln in data[1:1 + n]:
        p = ln.split()
        if len(p) >= 4:
            rows.append((float(p[0]), float(p[1]), float(p[2]), float(p[3])))
    umax = max(max(abs(u1), abs(u2)) for u1, u2, _, _ in rows)

    # coarse 2-D grid over (r, c)
    best = None
    cg_lo = umax + 0.02
    for ci in range(0, 26):
        c = cg_lo + ci * (4.0 - cg_lo) / 25.0
        for ri in range(0, 34):
            r = 0.15 + ri * (3.2 - 0.15) / 33.0
            rr = resid(rows, r, c)
            if best is None or rr < best[0]:
                best = (rr, r, c)
    _, r, c = best

    # local coordinate refinement (alternate golden searches on c then r)
    for _ in range(6):
        c = golden_min(lambda cc: resid(rows, r, cc), max(cg_lo, c - 0.4), c + 0.4)
        r = golden_min(lambda rr: resid(rows, rr, c), max(0.05, r - 0.4), r + 0.4)

    # Emit the physical (exchange) branch as a compact closed form:
    #   V  = centre-of-momentum velocity = (g1*u1 + g2*r*u2)/(g1 + g2*r)
    #   v1 = (2V - u1 - u1*V^2/c^2) / (1 + V^2/c^2 - 2*u1*V/c^2)
    # (the reflection of u1 about V under the ceiling-c velocity addition).
    C = repr(c)
    C2 = repr(c * c)
    R = repr(r)
    # G = g2/g1 keeps each gamma-radical to a single sqrt in V.
    G = "sqrt(abs((1-(u1/%s)**2)/(1-(u2/%s)**2)))" % (C, C)
    V = "((u1+%s*%s*u2)/(1+%s*%s))" % (G, R, G, R)
    num = "(2*%s-u1-u1*%s**2/%s)" % (V, V, C2)
    den = "(1+%s**2/%s-2*u1*%s/%s)" % (V, C2, V, C2)
    v1 = "(%s/%s)" % (num, den)
    sys.stdout.write(v1 + "\n")


if __name__ == "__main__":
    main()
