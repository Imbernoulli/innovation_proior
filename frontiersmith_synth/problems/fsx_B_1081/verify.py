#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- checker for fsx_B_1081
   Family: gore-strain-paneling (format C, minimize maximum flattening strain).

Instance: a triangulated heightfield "drum skin" (grid of vertices with 3D
coords (i, j, h(i,j)), two triangles per grid cell, fixed diagonal). The
solver partitions the N = 2*R*C triangles into connected PANELS (mechanism:
developable-panel-layout) under a total seam-length budget B (mechanism:
seam-length-cap). Score = the maximum, over all interior mesh vertices,
of "unrelieved" discrete Gaussian curvature (angle defect) (mechanism:
flattening-strain-metric): a vertex whose incident triangles are not ALL
in the same panel sits on a seam and is relieved (contributes 0); a vertex
fully surrounded by one panel cannot be isometrically flattened without
strain proportional to its angle defect (Gauss's Theorema Egregium).
"""
import sys, math

MAX_TOKENS = 6000


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


def build_mesh(R, C, h):
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
    return V, TRI


def edge_key(a, b):
    return (a, b) if a < b else (b, a)


def build_edge_tris(TRI):
    et = {}
    for idx, (a, b, c) in enumerate(TRI):
        for (p, q) in ((a, b), (b, c), (c, a)):
            et.setdefault(edge_key(p, q), []).append(idx)
    return et


def build_vertex_incident(TRI):
    vi = {}
    for idx, (a, b, c) in enumerate(TRI):
        for v in (a, b, c):
            vi.setdefault(v, []).append(idx)
    return vi


def compute_K(V, TRI, R, C):
    """Discrete Gaussian curvature (angle defect) at every FULL-interior grid
    vertex (1<=i<=R-1, 1<=j<=C-1 -- these have a complete 6-triangle fan, so
    the flat target angle sum is exactly 2*pi). Domain-boundary vertices are
    excluded (their target isn't 2*pi and they're always free, like the rim
    of the drum)."""
    angsum = {}
    for (a, b, c) in TRI:
        for (p, q, r) in ((a, b, c), (b, c, a), (c, a, b)):
            angsum[p] = angsum.get(p, 0.0) + angle_at(V[p], V[q], V[r])
    K = {}
    for i in range(1, R):
        for j in range(1, C):
            v = vid(i, j, C)
            K[v] = 2.0 * math.pi - angsum.get(v, 0.0)
    return K


def seam_length(V, edge_tris, panel):
    s = 0.0
    for (a, b), tris in edge_tris.items():
        if len(tris) == 2:
            t1, t2 = tris
            if panel[t1] != panel[t2]:
                s += norm(sub(V[a], V[b]))
    return s


def read_instance(path):
    toks = open(path).read().split()
    ptr = 0
    R = int(toks[ptr]); ptr += 1
    C = int(toks[ptr]); ptr += 1
    B = float(toks[ptr]); ptr += 1
    h = [[0.0] * (C + 1) for _ in range(R + 1)]
    for i in range(R + 1):
        for j in range(C + 1):
            h[i][j] = float(toks[ptr]); ptr += 1
    return R, C, B, h


def parse_panels(text, N):
    toks = text.split()
    if len(toks) == 0:
        return None, "empty output"
    if len(toks) != N:
        return None, f"expected exactly {N} tokens, got {len(toks)}"
    if len(toks) > MAX_TOKENS:
        return None, "too many tokens"
    panel = []
    for t in toks:
        try:
            v = int(t)
        except ValueError:
            return None, "non-integer token (nan/inf/garbage)"
        if v < 0 or v >= N:
            return None, f"panel id {v} out of range [0,{N-1}]"
        panel.append(v)
    return panel, "ok"


def check_connected(TRI, edge_tris, panel):
    """Union-Find over triangles; union across shared edges with equal panel id.
    Every used panel id's triangle set must land in ONE component."""
    n = len(TRI)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for (a, b), tris in edge_tris.items():
        if len(tris) == 2:
            t1, t2 = tris
            if panel[t1] == panel[t2]:
                union(t1, t2)

    groups = {}
    for idx in range(n):
        groups.setdefault(panel[idx], set()).add(find(idx))
    for pid, roots in groups.items():
        if len(roots) > 1:
            return False, pid
    return True, None


def evaluate(panel, V, TRI, edge_tris, vertex_incident, K, R, C):
    Fval = 0.0
    for v in K:
        ids = set(panel[t] for t in vertex_incident.get(v, []))
        if len(ids) <= 1:
            a = abs(K[v])
            if a > Fval:
                Fval = a
    return Fval


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0")
        return 0
    inf, outf = sys.argv[1], sys.argv[2]
    R, C, B, h = read_instance(inf)
    V, TRI = build_mesh(R, C, h)
    N = len(TRI)
    edge_tris = build_edge_tris(TRI)
    vertex_incident = build_vertex_incident(TRI)
    K = compute_K(V, TRI, R, C)

    text = open(outf).read()
    panel, reason = parse_panels(text, N)
    if panel is None:
        print(f"infeasible: {reason}")
        print("Ratio: 0.0")
        return 0

    conn_ok, bad_pid = check_connected(TRI, edge_tris, panel)
    if not conn_ok:
        print(f"infeasible: panel {bad_pid} is not a single connected region")
        print("Ratio: 0.0")
        return 0

    S = seam_length(V, edge_tris, panel)
    if S > B + 1e-6:
        print(f"infeasible: seam length {S:.6f} exceeds budget {B:.6f}")
        print("Ratio: 0.0")
        return 0

    Fval = evaluate(panel, V, TRI, edge_tris, vertex_incident, K, R, C)
    if not math.isfinite(Fval):
        print("non-finite objective")
        print("Ratio: 0.0")
        return 0

    base_panel = [0] * N
    Fbase = evaluate(base_panel, V, TRI, edge_tris, vertex_incident, K, R, C)
    Fbase = max(Fbase, 1e-6)

    sc = min(1000.0, 100.0 * Fbase / max(1e-9, Fval))
    print(f"seam={S:.4f}/{B:.4f} F={Fval:.6f} Fbase={Fbase:.6f}")
    print("Ratio: %.6f" % (sc / 1000.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
