#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for cospectral-decoy-mints.

Feasibility (any violation -> "Ratio: 0.0"):
  - participant prints two simple undirected labeled graphs G, H on exactly the n
    vertices given in the instance (1..n), each with edge count <= M, no self-loops,
    no duplicate edges;
  - G and H must be exactly cospectral: sorted adjacency eigenvalues agree within 1e-6.

Objective (maximize): F(G,H) = |sum-of-component-diameters(G) - sum-of-component-diameters(H)|
  + (1/n) * L1(sorted degree sequence of G, sorted degree sequence of H).

Score: checker builds its own small reference pair (G0,H0) -- a single localized
switch of one 6-vertex gadget block inside an otherwise-uniform disjoint union -- and
reports Ratio = min(1, F(G,H) / (10 * F(G0,H0))).
"""
import sys
import numpy as np

TOL = 1e-6

# A verified (numerically, to ~1e-13) non-isomorphic cospectral mate pair on 6 vertices:
# degree sequences (1,2,2,2,2,5) vs (1,1,3,3,3,3); component diameters 2 vs 4.
GADGET_A = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 4), (2, 3)]
GADGET_B = [(0, 2), (0, 3), (0, 5), (1, 2), (1, 3), (1, 4), (2, 3)]
GADGET_SIZE = 6


def fail(msg):
    print("INVALID: %s" % msg)
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    n = int(toks[0]); m = int(toks[1])
    return n, m


def build_adj(n, edges):
    A = np.zeros((n, n), dtype=float)
    for u, v in edges:
        A[u - 1, v - 1] = 1.0
        A[v - 1, u - 1] = 1.0
    return A


def adj_list(n, edges):
    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def component_diam_sum(n, edges):
    adj = adj_list(n, edges)
    visited = [False] * (n + 1)
    total = 0
    for s in range(1, n + 1):
        if visited[s]:
            continue
        comp = [s]
        visited[s] = True
        stack = [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
                    comp.append(v)
        if len(comp) <= 1:
            continue
        diam = 0
        for u in comp:
            dist = {u: 0}
            q = [u]
            qi = 0
            while qi < len(q):
                cur = q[qi]; qi += 1
                for v in adj[cur]:
                    if v not in dist:
                        dist[v] = dist[cur] + 1
                        q.append(v)
            m = max(dist.values())
            if m > diam:
                diam = m
        total += diam
    return total


def divergence(n, edgesG, edgesH):
    dG = component_diam_sum(n, edgesG)
    dH = component_diam_sum(n, edgesH)
    degG = [0] * (n + 1)
    degH = [0] * (n + 1)
    for u, v in edgesG:
        degG[u] += 1; degG[v] += 1
    for u, v in edgesH:
        degH[u] += 1; degH[v] += 1
    sdegG = sorted(degG[1:])
    sdegH = sorted(degH[1:])
    l1 = sum(abs(a - b) for a, b in zip(sdegG, sdegH))
    return abs(dG - dH) + l1 / n


def parse_graph(toks, pos, n, m):
    """Parse one 'n1 e1' + e1 edges block starting at toks[pos]. Returns (edges, new_pos)."""
    n1 = int(toks[pos]); pos += 1
    e1 = int(toks[pos]); pos += 1
    if n1 != n:
        fail("graph vertex count %d != n=%d" % (n1, n))
    if e1 < 0 or e1 > m:
        fail("edge count %d out of [0,%d]" % (e1, m))
    edges = []
    seen = set()
    for _ in range(e1):
        u = int(toks[pos]); pos += 1
        v = int(toks[pos]); pos += 1
        if u < 1 or u > n or v < 1 or v > n:
            fail("vertex id out of range")
        if u == v:
            fail("self-loop at %d" % u)
        key = (u, v) if u < v else (v, u)
        if key in seen:
            fail("duplicate edge (%d,%d)" % (u, v))
        seen.add(key)
        edges.append((u, v))
    return edges, pos


def build_gadget_baseline(n):
    """Reference pair: k copies of GADGET_A (=G0), with exactly ONE block replaced by
    GADGET_B (=H0). n is guaranteed to be a multiple of GADGET_SIZE by gen.py."""
    k = n // GADGET_SIZE
    edgesG = []
    edgesH = []
    for i in range(k):
        base = i * GADGET_SIZE
        for (a, b) in GADGET_A:
            edgesG.append((base + a + 1, base + b + 1))
        variant = GADGET_B if i == 0 else GADGET_A
        for (a, b) in variant:
            edgesH.append((base + a + 1, base + b + 1))
    return edgesG, edgesH


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0")
        return
    in_path, out_path = sys.argv[1], sys.argv[2]
    n, m = read_instance(in_path)

    try:
        with open(out_path) as f:
            raw = f.read()
        toks = raw.split()
    except Exception:
        fail("cannot read output")

    try:
        pos = 0
        edgesG, pos = parse_graph(toks, pos, n, m)
        edgesH, pos = parse_graph(toks, pos, n, m)
    except (ValueError, IndexError):
        fail("malformed / truncated output")

    # finiteness / sane-range guard (defensive; parse_graph already used int()).
    for (u, v) in edgesG + edgesH:
        if not (isinstance(u, int) and isinstance(v, int)):
            fail("non-finite token")

    AG = build_adj(n, edgesG)
    AH = build_adj(n, edgesH)
    try:
        evG = np.sort(np.linalg.eigvalsh(AG))
        evH = np.sort(np.linalg.eigvalsh(AH))
    except np.linalg.LinAlgError:
        fail("eigendecomposition failed")

    if not (np.all(np.isfinite(evG)) and np.all(np.isfinite(evH))):
        fail("non-finite spectrum")

    max_diff = float(np.max(np.abs(evG - evH)))
    if max_diff > TOL:
        fail("not cospectral (max eigenvalue gap %.3e > %.1e)" % (max_diff, TOL))

    F = divergence(n, edgesG, edgesH)

    base_edgesG, base_edgesH = build_gadget_baseline(n)
    B = divergence(n, base_edgesG, base_edgesH)
    B = max(1e-9, B)

    sc = min(1000.0, 100.0 * F / B)
    print("F=%.6f B=%.6f maxEigGap=%.3e" % (F, B, max_diff))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
