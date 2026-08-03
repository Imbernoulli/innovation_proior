# TIER: strong
# Bottleneck-driven local search on the triangular plate. Build several
# structured + jittered starts (jittered rings, jittered triangular lattices,
# random) to avoid the symmetry lock of a perfect ring, pick the best, then
# repeatedly locate the smallest-area triple and either nudge or relocate one
# of its three pads (always projected back into the triangle), accepting only
# moves that raise the GLOBAL minimum triangle area. Seeded => deterministic.
import sys, math, random

RING_R = 0.15
CX, CY = 1.0 / 3.0, 1.0 / 3.0


def tri_area(p, q, r):
    return 0.5 * abs((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))


def eval_config(pts):
    """Return (min_area, (i,j,k)) of the bottleneck triple."""
    n = len(pts)
    best = float("inf")
    tri = (0, 1, 2)
    for a in range(n):
        pa = pts[a]
        for b in range(a + 1, n):
            pb = pts[b]
            for c in range(b + 1, n):
                ar = tri_area(pa, pb, pts[c])
                if ar < best:
                    best = ar
                    tri = (a, b, c)
    return best, tri


def proj_tri(x, y):
    """Project (x,y) into the unit right triangle {x>=0, y>=0, x+y<=1}."""
    if x < 0.0:
        x = 0.0
    if y < 0.0:
        y = 0.0
    s = x + y
    if s > 1.0:
        # pull back along the direction toward the origin
        x /= s
        y /= s
    return x, y


def rand_pt(rng):
    x = rng.random()
    y = rng.random()
    if x + y > 1.0:
        x, y = 1.0 - x, 1.0 - y
    return [x, y]


def ring(N):
    return [list(proj_tri(CX + RING_R * math.cos(2 * math.pi * k / N),
                          CY + RING_R * math.sin(2 * math.pi * k / N)))
            for k in range(N)]


def jittered_ring(N, rng, amp):
    return [list(proj_tri(x + rng.uniform(-amp, amp), y + rng.uniform(-amp, amp)))
            for (x, y) in ring(N)]


def jittered_lattice(N, rng, amp):
    """Triangular-grid-ish lattice: barycentric-style points on the plate."""
    m = 2
    while (m + 1) * (m + 2) // 2 < N:
        m += 1
    pts = []
    for i in range(m + 1):
        for j in range(m + 1 - i):
            if len(pts) >= N:
                break
            x = i / float(m)
            y = j / float(m)
            pts.append(list(proj_tri(x + rng.uniform(-amp, amp),
                                     y + rng.uniform(-amp, amp))))
    while len(pts) < N:
        pts.append(rand_pt(rng))
    return pts[:N]


def main():
    N = int(sys.stdin.read().split()[0])
    rng = random.Random(3131 + N)

    # ---- candidate starts: jittered / lattice / random only (NEVER the exact
    # symmetric ring, which is symmetry-locked under single-vertex moves) ----
    best_pts, best_val = None, -1.0
    for _ in range(20):
        for cand in (jittered_ring(N, rng, 0.10),
                     jittered_lattice(N, rng, 0.08),
                     [rand_pt(rng) for _ in range(N)]):
            v = eval_config(cand)[0]
            if v > best_val:
                best_val, best_pts = v, [list(p) for p in cand]

    cur = [[p[0], p[1]] for p in best_pts]
    cur_val, tri = eval_config(cur)

    # ---- bottleneck-driven local search: nudge or relocate ----
    step = 0.25
    iters = 6000
    for it in range(iters):
        if it and it % 800 == 0:
            step *= 0.7
        vtx = tri[rng.randrange(3)]
        ox, oy = cur[vtx][0], cur[vtx][1]
        if rng.random() < 0.5:
            nx, ny = proj_tri(ox + rng.uniform(-step, step),
                              oy + rng.uniform(-step, step))
        else:
            p = rand_pt(rng)
            nx, ny = p[0], p[1]
        cur[vtx][0], cur[vtx][1] = nx, ny
        nv, ntri = eval_config(cur)
        if nv > cur_val + 1e-15:
            cur_val, tri = nv, ntri
        else:
            cur[vtx][0], cur[vtx][1] = ox, oy

    out = ["%.12f %.12f" % (p[0], p[1]) for p in cur]
    sys.stdout.write("\n".join(out) + "\n")


main()
