#!/usr/bin/env python3
"""gen.py <testId> -> prints ONE embroidery stitch-plan instance to stdout.

Instance schema:
  line 1: V E C K          vertices, stitch segments, colors, color-change cost
  next V lines: x y        integer grid coordinates of each vertex (0-based id)
  next E lines: u v c      unit stitch segment between vertices u,v with color c

The needle starts parked at vertex 0 with NO color loaded. Ops (for the solver):
  STITCH an unstitched segment adjacent to the needle: cost 1 (+K if its color
  differs from the currently loaded color; loading the first color also costs K).
  JUMP the needle to any vertex: cost = Manhattan distance.

Planted structure (the trap): the design is a set of R congruent copies of one
multicolor "weave" motif, placed on a wide translation lattice with jitter --
the symmetry group is the translation group of the lattice, and the motif copies
are its orbit. Each motif copy carries ALL C colors in diagonal bands, so each
color class is fragmented into several arcs per region. Edges are listed region
by region in a SCRAMBLED region order (not a short tour) and in scan order
within a region, so the input-order baseline churns color changes and makes
long jumps. Sorting all segments by color (the obvious recipe) minimizes color
changes but sweeps the whole lattice once per color -- the orbit structure makes
re-buying color changes along ONE short tour of the regions strictly cheaper.

Difficulty ladder 1..10 grows the lattice, motif size, color count and K.
Deterministic: seeded only by testId.
"""
import sys
import random

LADDER = {
    1: dict(a=2, b=2, D=60, m=8, w=2, C=3, K=14, jit=4, flip=0),
    2: dict(a=2, b=3, D=70, m=8, w=2, C=3, K=16, jit=5, flip=1),
    3: dict(a=3, b=3, D=80, m=9, w=2, C=4, K=16, jit=6, flip=0),
    4: dict(a=3, b=3, D=100, m=10, w=2, C=4, K=18, jit=8, flip=1),
    5: dict(a=3, b=3, D=110, m=10, w=3, C=4, K=20, jit=8, flip=0),
    6: dict(a=3, b=4, D=120, m=10, w=2, C=4, K=20, jit=10, flip=1),
    7: dict(a=3, b=4, D=130, m=11, w=3, C=5, K=20, jit=10, flip=0),
    8: dict(a=4, b=4, D=140, m=11, w=2, C=5, K=22, jit=12, flip=1),
    9: dict(a=4, b=4, D=150, m=12, w=3, C=5, K=24, jit=12, flip=0),
    10: dict(a=4, b=4, D=160, m=12, w=2, C=6, K=24, jit=14, flip=1),
}


def motif_edges(m, w, C, flip):
    """The weave motif: horizontal unit edges on even local rows, vertical unit
    edges on even local columns. Color = diagonal band ((lx+ly)//w) % C of the
    anchor (lower-left) endpoint. Returns (vertices, edges) in local coords;
    vertices are those touched by >=1 edge; edges are (lx, ly, dir, color),
    dir 0 = horizontal (lx,ly)-(lx+1,ly), dir 1 = vertical (lx,ly)-(lx,ly+1).
    """
    used = set()
    edges = []
    for ly in range(m):
        for lx in range(m):
            if ly % 2 == 0 and lx + 1 < m:  # horizontal
                col = ((lx + ly) // w) % C
                edges.append((lx, ly, 0, col))
                used.add((lx, ly)); used.add((lx + 1, ly))
            if lx % 2 == 0 and ly + 1 < m:  # vertical
                col = ((lx + ly) // w) % C
                edges.append((lx, ly, 1, col))
                used.add((lx, ly)); used.add((lx, ly + 1))
    if flip:  # transpose the motif (still the same orbit representative shape)
        edges = [(ly, lx, 1 - d, c) for (lx, ly, d, c) in edges]
        used = {(y, x) for (x, y) in used}
    return sorted(used), sorted(edges, key=lambda e: (e[1], e[0], e[2]))


def main():
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if tid not in LADDER:
        tid = ((tid - 1) % 10) + 1
    p = LADDER[tid]
    rng = random.Random(911000 + tid)
    a, b, D, m, w, C, K, jit, flip = (p[k] for k in
                                     ("a", "b", "D", "m", "w", "C", "K", "jit", "flip"))

    verts_local, edges_local = motif_edges(m, w, C, flip)

    # placement: a x b lattice, spacing D, per-region jitter; regions stay far
    # apart (D >= m + 2*jit + 4 on every rung) and never share vertices.
    regions = [(i, j) for i in range(a) for j in range(b)]
    origin = {}
    for (i, j) in regions:
        ox = i * D + rng.randint(-jit, jit)
        oy = j * D + rng.randint(-jit, jit)
        origin[(i, j)] = (ox + jit, oy + jit)  # shift so all coords >= 0

    # global vertex ids assigned in placement order, local scan order
    vid = {}
    coords = []
    for (i, j) in regions:
        ox, oy = origin[(i, j)]
        for (lx, ly) in verts_local:
            vid[((i, j), lx, ly)] = len(coords)
            coords.append((ox + lx, oy + ly))

    # edge listing: regions in a SCRAMBLED order (not a tour); within a region,
    # scan order (ly, lx, dir) which alternates colors at band boundaries.
    order = list(regions)
    rng.shuffle(order)
    out_edges = []
    for rg in order:
        ox, oy = origin[rg]
        for (lx, ly, d, c) in edges_local:
            if d == 0:
                u = vid[(rg, lx, ly)]; v = vid[(rg, lx + 1, ly)]
            else:
                u = vid[(rg, lx, ly)]; v = vid[(rg, lx, ly + 1)]
            out_edges.append((u, v, c))

    wout = sys.stdout.write
    wout("%d %d %d %d\n" % (len(coords), len(out_edges), C, K))
    for (x, y) in coords:
        wout("%d %d\n" % (x, y))
    for (u, v, c) in out_edges:
        wout("%d %d %d\n" % (u, v, c))


if __name__ == "__main__":
    main()
