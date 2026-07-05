# TIER: trivial
# Weak reference: string the single best triangular loop over 3 stations.
# Captures at most one formation cluster and pays a triangle's rope -> ~0.1.
import sys, json, math


def orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def pt_in_tri(p, a, b, c):
    d1 = orient(a, b, p); d2 = orient(b, c, p); d3 = orient(c, a, p)
    return (d1 > 0 and d2 > 0 and d3 > 0) or (d1 < 0 and d2 < 0 and d3 < 0)


def perim(V):
    s = 0.0
    n = len(V)
    for i in range(n):
        ax, ay = V[i]; bx, by = V[(i + 1) % n]
        s += math.hypot(ax - bx, ay - by)
    return s


def main():
    inst = json.load(sys.stdin)
    st = inst["stations"]; feats = inst["features"]; lam = inst["lam"]
    n = len(st)
    best_val = None; best = (0, 1, 2)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = st[i], st[j], st[k]
                if orient(a, b, c) == 0:
                    continue
                val = 0
                for (fx, fy, fv) in feats:
                    if pt_in_tri((fx, fy), a, b, c):
                        val += fv
                v = val - lam * perim([a, b, c])
                if best_val is None or v > best_val:
                    best_val = v; best = (i, j, k)
    print(json.dumps({"tour": list(best)}))


main()
