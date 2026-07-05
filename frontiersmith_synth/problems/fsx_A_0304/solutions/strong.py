# TIER: strong
# Bottleneck-driven local search over several seeded restarts. Each restart
# starts from a structured layout (boundary ring + interior jittered grid),
# then repeatedly locates the smallest-area triple and perturbs one of its
# vertices, accepting any move that raises the GLOBAL minimum triangle area.
# Deterministic (seed derived from N only).
import sys, random, math


def worst_triple(pts):
    n = len(pts)
    best = float("inf")
    tri = (0, 1, 2)
    for a in range(n):
        xa, ya = pts[a]
        for b in range(a + 1, n):
            xb, yb = pts[b]
            dx1 = xb - xa
            dy1 = yb - ya
            for c in range(b + 1, n):
                xc, yc = pts[c]
                area = 0.5 * abs(dx1 * (yc - ya) - dy1 * (xc - xa))
                if area < best:
                    best = area
                    tri = (a, b, c)
    return best, tri


def clamp01(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def structured_init(N, rng, jit):
    # Place ceil-fraction of points on the boundary ring, rest on an interior grid.
    pts = []
    nb = min(N, max(4, N // 2))
    for k in range(nb):
        # walk the square perimeter (perimeter length 4)
        s = 4.0 * k / nb
        if s < 1.0:
            x, y = s, 0.0
        elif s < 2.0:
            x, y = 1.0, s - 1.0
        elif s < 3.0:
            x, y = 3.0 - s, 1.0
        else:
            x, y = 0.0, 4.0 - s
        pts.append((clamp01(x + rng.uniform(-jit, jit)),
                    clamp01(y + rng.uniform(-jit, jit))))
    rem = N - nb
    if rem > 0:
        g = int(math.ceil(math.sqrt(rem)))
        placed = 0
        for i in range(g):
            for j in range(g):
                if placed >= rem:
                    break
                x = (i + 1.0) / (g + 1.0)
                y = (j + 1.0) / (g + 1.0)
                pts.append((clamp01(x + rng.uniform(-jit, jit)),
                            clamp01(y + rng.uniform(-jit, jit))))
                placed += 1
    return pts


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    rng = random.Random(7 * N + 3)

    # Seed the search with the full-incircle ring so strong is never worse than
    # the greedy ring construction.
    cx, cy, rr = 0.5, 0.5, 0.5
    ring = [(cx + rr * math.cos(2.0 * math.pi * k / N),
             cy + rr * math.sin(2.0 * math.pi * k / N)) for k in range(N)]
    best_global_pts = list(ring)
    best_global_val, _ = worst_triple(ring)

    restarts = 6
    iters = 1200
    for r in range(restarts):
        if r == 0:
            pts = list(ring)
        else:
            pts = structured_init(N, rng, 0.06 + 0.02 * r)
        cur_val, _ = worst_triple(pts)
        step = 0.25
        for it in range(iters):
            _, tri = worst_triple(pts)
            idx = tri[rng.randrange(3)]
            ox, oy = pts[idx]
            nx = clamp01(ox + rng.uniform(-step, step))
            ny = clamp01(oy + rng.uniform(-step, step))
            pts[idx] = (nx, ny)
            nv, _ = worst_triple(pts)
            if nv > cur_val:
                cur_val = nv
            else:
                pts[idx] = (ox, oy)
            if it % 200 == 199:
                step *= 0.6
        if cur_val > best_global_val:
            best_global_val = cur_val
            best_global_pts = list(pts)

    sys.stdout.write("\n".join("%.12f %.12f" % p for p in best_global_pts) + "\n")


if __name__ == "__main__":
    main()
