# TIER: strong
# Multi-restart annealed relaxation. From several seeded initial layouts (an incremental
# init plus random restarts) it repeatedly attacks the BINDING (thinnest) wake triangle:
# it tries seeded random repositions of one of that triangle's vertices and keeps any move
# that enlarges the overall minimum triangle area. Best layout over all restarts is emitted.
import sys, math, random

LO, HI = 0.02, 0.98

def tri_area(a, b, c):
    return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

def full_min(pts):
    m = len(pts)
    best = None; bt = (0, 1, 2)
    for i in range(m):
        ai = pts[i]
        for j in range(i + 1, m):
            aj = pts[j]
            for k in range(j + 1, m):
                a = tri_area(ai, aj, pts[k])
                if best is None or a < best:
                    best = a; bt = (i, j, k)
    return (0.0 if best is None else best), bt

def min_excluding(pts, v):
    m = len(pts)
    best = None
    for i in range(m):
        if i == v:
            continue
        ai = pts[i]
        for j in range(i + 1, m):
            if j == v:
                continue
            aj = pts[j]
            for k in range(j + 1, m):
                if k == v:
                    continue
                a = tri_area(ai, aj, pts[k])
                if best is None or a < best:
                    best = a
    return best if best is not None else float("inf")

def min_containing_at(pts, v, cand):
    m = len(pts)
    best = None
    for i in range(m):
        if i == v:
            continue
        ai = pts[i]
        for j in range(i + 1, m):
            if j == v:
                continue
            a = tri_area(ai, pts[j], cand)
            if best is None or a < best:
                best = a
    return best if best is not None else float("inf")

def clamp(x):
    return LO if x < LO else (HI if x > HI else x)

def greedy_init(n):
    def min_pair_area(pts, cand):
        m = len(pts); best = None
        for i in range(m):
            ai = pts[i]
            for j in range(i + 1, m):
                a = tri_area(ai, pts[j], cand)
                if best is None or a < best:
                    best = a
        return best if best is not None else float("inf")
    def dist2_min(pts, cand):
        best = None
        for p in pts:
            d = (p[0] - cand[0]) ** 2 + (p[1] - cand[1]) ** 2
            if best is None or d < best:
                best = d
        return best if best is not None else float("inf")
    G = 18
    grid = [(LO + (HI - LO) * a / (G - 1), LO + (HI - LO) * b / (G - 1))
            for a in range(G) for b in range(G)]
    pts = []
    for _ in range(n):
        best_c = None; best_val = -1.0
        for c in grid:
            val = dist2_min(pts, c) if len(pts) < 2 else min_pair_area(pts, c)
            if val > best_val:
                best_val = val; best_c = c
        pts.append(list(best_c))
    return pts

def random_init(n, rng):
    return [[LO + (HI - LO) * rng.random(), LO + (HI - LO) * rng.random()] for _ in range(n)]

def refine(pts, rng, rounds, T):
    f, _ = full_min(pts)
    for _ in range(rounds):
        f, (a, b, c) = full_min(pts)
        verts = [a, b, c]; rng.shuffle(verts)
        moved = False
        for v in verts:
            fex = min_excluding(pts, v)
            ox, oy = pts[v]
            best_val = f; best_pos = None
            for t in range(T):
                if t < T - 6:
                    sigma = 0.05 + 0.13 * rng.random()
                    cand = (clamp(ox + rng.gauss(0.0, sigma)), clamp(oy + rng.gauss(0.0, sigma)))
                else:
                    cand = (LO + (HI - LO) * rng.random(), LO + (HI - LO) * rng.random())
                gp = min_containing_at(pts, v, cand)
                val = gp if gp < fex else fex
                if val > best_val + 1e-15:
                    best_val = val; best_pos = cand
            if best_pos is not None:
                pts[v][0], pts[v][1] = best_pos; moved = True
                break
        if not moved:
            w = rng.randrange(len(pts))
            ox, oy = pts[w]
            pts[w][0] = clamp(ox + rng.gauss(0.0, 0.04)); pts[w][1] = clamp(oy + rng.gauss(0.0, 0.04))
            nf, _ = full_min(pts)
            if nf < f:
                pts[w][0], pts[w][1] = ox, oy
    return pts

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    rng = random.Random(1234567 + 7 * n)

    inits = [greedy_init(n)]
    for _ in range(3):
        inits.append(random_init(n, rng))

    best_pts = None; best_f = -1.0
    for k, init in enumerate(inits):
        rounds = 150 if k == 0 else 110
        pts = refine([list(p) for p in init], rng, rounds, T=26)
        f, _ = full_min(pts)
        if f > best_f:
            best_f = f; best_pts = pts

    out = ["%.17g %.17g" % (p[0], p[1]) for p in best_pts]
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
