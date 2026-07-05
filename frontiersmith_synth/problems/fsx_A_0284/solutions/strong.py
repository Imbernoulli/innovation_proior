# TIER: strong
# Bottleneck-driven local search. Build several structured + jittered starts
# (jittered rings, jittered grids, random) to avoid the symmetry lock of a
# perfect ring, pick the best, then repeatedly locate the smallest-area triple
# and either nudge or relocate one of its three towers, accepting only moves
# that raise the GLOBAL minimum triangle area. Seeded => deterministic.
import sys, math, random

RING_R = 0.20
CX, CY = 0.5, 0.5


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


def clamp01(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def ring(N):
    return [[CX + RING_R * math.cos(2 * math.pi * k / N),
             CY + RING_R * math.sin(2 * math.pi * k / N)] for k in range(N)]


def jittered_ring(N, rng, amp):
    return [[clamp01(x + rng.uniform(-amp, amp)),
             clamp01(y + rng.uniform(-amp, amp))] for (x, y) in ring(N)]


def jittered_grid(N, rng, amp):
    m = int(math.ceil(math.sqrt(N)))
    pts = []
    for r in range(m):
        for c in range(m):
            if len(pts) >= N:
                break
            x = (c + 0.5) / m
            y = (r + 0.5) / m
            pts.append([clamp01(x + rng.uniform(-amp, amp)),
                        clamp01(y + rng.uniform(-amp, amp))])
    return pts[:N]


def main():
    N = int(sys.stdin.read().split()[0])
    rng = random.Random(4242 + N)

    # ---- candidate starts: jittered / grid / random only (NEVER the exact
    # symmetric ring, which is symmetry-locked under single-vertex moves) ----
    best_pts, best_val = None, -1.0
    for _ in range(20):
        for cand in (jittered_ring(N, rng, 0.12),
                     jittered_grid(N, rng, 0.10),
                     [[rng.random(), rng.random()] for _ in range(N)]):
            v = eval_config(cand)[0]
            if v > best_val:
                best_val, best_pts = v, cand

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
            cur[vtx][0] = clamp01(ox + rng.uniform(-step, step))
            cur[vtx][1] = clamp01(oy + rng.uniform(-step, step))
        else:
            cur[vtx][0] = rng.random()
            cur[vtx][1] = rng.random()
        nv, ntri = eval_config(cur)
        if nv > cur_val + 1e-15:
            cur_val, tri = nv, ntri
        else:
            cur[vtx][0], cur[vtx][1] = ox, oy

    out = ["%.12f %.12f" % (p[0], p[1]) for p in cur]
    sys.stdout.write("\n".join(out) + "\n")


main()
