# TIER: strong
# Anchors + free points chosen by trying several low-discrepancy generators
# (Hammersley, Fibonacci/golden-ratio lattices) under a few shifts, then
# keeping whichever full set has the smallest EXACT star discrepancy.
import sys

def vdc2(k):
    v = 0.0; denom = 0.5
    while k > 0:
        v += (k & 1) * denom
        k >>= 1
        denom *= 0.5
    return v

def star_discrepancy(points):
    n = len(points)
    if n == 0:
        return 1.0
    xs = sorted(set(p[0] for p in points) | {1.0})
    ys = sorted(set(p[1] for p in points) | {1.0})
    disc = 0.0
    for a in xs:
        for b in ys:
            vol = a * b
            cl = 0; op = 0
            for (px, py) in points:
                if px <= a and py <= b:
                    cl += 1
                if px < a and py < b:
                    op += 1
            v1 = cl / n - vol
            v2 = vol - op / n
            if v1 > disc: disc = v1
            if v2 > disc: disc = v2
    return disc

def gen_free(rem, kind, sx, sy):
    phi = 0.6180339887498949
    pts = []
    for j in range(rem):
        u = (j + 0.5) / rem if rem > 0 else 0.5
        if kind == "hammersley":
            x, y = u, vdc2(j)
        elif kind == "fib":
            x, y = u, (j * phi) % 1.0
        elif kind == "fib2":
            x, y = (j * phi) % 1.0, u
        else:  # grid-ish jitter
            x, y = u, ((j * 3) % max(1, rem)) / max(1, rem)
        pts.append(((x + sx) % 1.0, (y + sy) % 1.0))
    return pts

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    dim = int(next(it)); M = int(next(it)); K = int(next(it))
    anchors = [(float(next(it)), float(next(it))) for _ in range(K)]
    rem = M - K

    best = None; best_d = None
    shifts = [0.0, 0.5, 0.25, 0.125]
    for kind in ("hammersley", "fib", "fib2", "grid"):
        for sx in shifts:
            for sy in shifts:
                free = gen_free(rem, kind, sx, sy)
                full = anchors + free
                dd = star_discrepancy(full)
                if best_d is None or dd < best_d:
                    best_d = dd; best = free
    pts = anchors + (best if best is not None else [])
    print("\n".join("%.6f %.6f" % (x, y) for (x, y) in pts))

if __name__ == "__main__":
    main()
