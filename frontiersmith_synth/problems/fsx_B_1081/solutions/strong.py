# TIER: strong
"""The insight: seams are a budgeted resource for absorbing Gaussian
curvature, so spend them where curvature actually concentrates, not in any
extrinsic symmetric pattern. Compute the discrete angle defect (Gaussian
curvature) at every full-interior vertex directly from the 3D geometry,
rank vertices by |curvature| descending, and greedily carve small panels
around the highest-curvature vertices FIRST -- merging a vertex into an
ALREADY-carved neighboring panel when possible (clustering nearby peaks
into one cheap shared boundary) instead of always paying for a fresh cut.
Skip a candidate outright if isolating it would blow the seam budget, and
keep going down the ranked list (so a handful of cheap nearby peaks are not
blocked by one expensive outlier). This directly exploits the innovation
hook: track the curvature field, don't follow a regular pattern."""
import math
import sys


def vid(i, j, C):
    return i * (C + 1) + j


def sub(p, q):
    return (p[0] - q[0], p[1] - q[1], p[2] - q[2])


def dot(p, q):
    return p[0] * q[0] + p[1] * q[1] + p[2] * q[2]


def norm(p):
    return math.sqrt(dot(p, p))


def angle_at(A, B, C):
    u = sub(B, A); v = sub(C, A)
    nu = norm(u); nv = norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    c = dot(u, v) / (nu * nv)
    c = max(-1.0, min(1.0, c))
    return math.acos(c)


def edge_key(a, b):
    return (a, b) if a < b else (b, a)


def main():
    data = sys.stdin.read().split()
    ptr = 0
    R = int(data[ptr]); ptr += 1
    C = int(data[ptr]); ptr += 1
    B = float(data[ptr]); ptr += 1
    h = [[0.0] * (C + 1) for _ in range(R + 1)]
    for i in range(R + 1):
        for j in range(C + 1):
            h[i][j] = float(data[ptr]); ptr += 1

    V = []
    for i in range(R + 1):
        for j in range(C + 1):
            V.append((float(i), float(j), h[i][j]))
    TRI = []
    for i in range(R):
        for j in range(C):
            a = vid(i, j, C); b = vid(i + 1, j, C); c = vid(i + 1, j + 1, C); d = vid(i, j + 1, C)
            TRI.append((a, b, c))
            TRI.append((a, c, d))
    N = len(TRI)

    edge_tris = {}
    for idx, (a, b, c) in enumerate(TRI):
        for (p, q) in ((a, b), (b, c), (c, a)):
            edge_tris.setdefault(edge_key(p, q), []).append(idx)

    vertex_incident = {}
    for idx, (a, b, c) in enumerate(TRI):
        for v in (a, b, c):
            vertex_incident.setdefault(v, []).append(idx)

    angsum = {}
    for (a, b, c) in TRI:
        for (p, q, r) in ((a, b, c), (b, c, a), (c, a, b)):
            angsum[p] = angsum.get(p, 0.0) + angle_at(V[p], V[q], V[r])

    K = {}
    for i in range(1, R):
        for j in range(1, C):
            v = vid(i, j, C)
            K[v] = 2.0 * math.pi - angsum.get(v, 0.0)

    def seam_length(panel):
        s = 0.0
        for (a, b), tris in edge_tris.items():
            if len(tris) == 2:
                t1, t2 = tris
                if panel[t1] != panel[t2]:
                    s += norm(sub(V[a], V[b]))
        return s

    # neighbor-across-edge lookup per triangle (for merge-target search)
    tri_neighbors = [[] for _ in range(N)]
    for (a, b), tris in edge_tris.items():
        if len(tris) == 2:
            t1, t2 = tris
            tri_neighbors[t1].append(t2)
            tri_neighbors[t2].append(t1)

    panel = [0] * N
    next_id = 1

    def relieved(v):
        inc = vertex_incident.get(v, [])
        ids = set(panel[t] for t in inc)
        return len(ids) > 1

    ranked = sorted(K.keys(), key=lambda v: (-abs(K[v]), v))
    M = min(len(ranked), 60)
    for v in ranked[:M]:
        if relieved(v):
            continue
        inc = vertex_incident.get(v, [])
        if not inc:
            continue
        home = panel[inc[0]]  # all 6 currently share one id
        # flip exactly ONE of v's incident triangles so v touches >=2 ids.
        # try every candidate triangle t in v's fan, and for each, every
        # candidate target id (an already-carved neighbor's id, to cluster
        # cheaply, or a brand-new id) -- keep the option with the smallest
        # resulting total seam length that still fits the budget.
        best = None  # (new_seam, t, target)
        for t in inc:
            # candidate targets for THIS triangle: an id already carried by one
            # of t's own edge-neighbors (so the merge stays edge-connected), or
            # a brand-new singleton id (always trivially connected on its own).
            targets = {next_id}
            for nb in tri_neighbors[t]:
                if panel[nb] != home:
                    targets.add(panel[nb])
            for target in targets:
                if target == home:
                    continue
                old = panel[t]
                panel[t] = target
                s = seam_length(panel)
                panel[t] = old
                if s <= B + 1e-9 and (best is None or s < best[0]):
                    best = (s, t, target)
        if best is not None:
            _, t, target = best
            panel[t] = target
            if target == next_id:
                next_id += 1

    sys.stdout.write(" ".join(str(p) for p in panel) + "\n")


if __name__ == "__main__":
    main()
