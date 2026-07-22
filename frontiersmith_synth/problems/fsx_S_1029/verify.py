#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  -- deterministic checker for "Sprawl Doppelganger".

Feasibility (all exact integer / rational arithmetic, no floating point):
  - N,M parse as non-negative integers, 1 <= N <= n, 0 <= M <= M_MAX
  - every edge is a pair of distinct vertex indices in [0,N), no duplicate
    (undirected) edges, no self-loops
  - every vertex degree <= D_MAX
  - G is connected
  - for j=1..k: |mu_j(G) - mu_j(H)| <= eps   (mu_j = closed walks of length j / N)
  - |mu_S2(G) - mu_S2(H)| <= eps2            (mu_S2 = sum of squared degrees / N)
Any violation -> "Ratio: 0.0".

Objective (maximize): diam(G), the graph diameter.
Baseline B = diam(H) (H is itself always a feasible answer, L=1 of the same
covering-lift family), computed from the template edges present in <in>.
Score = min(1.0, diam(G) / (10*B)).
"""
import sys
import math
from collections import defaultdict, deque
from fractions import Fraction


def bfs_dist(n, adj, src):
    dist = [-1] * n
    dist[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        for x in adj[u]:
            if dist[x] == -1:
                dist[x] = dist[u] + 1
                q.append(x)
    return dist


def graph_diameter(n, adj):
    d = 0
    for v in range(n):
        dist = bfs_dist(n, adj, v)
        if -1 in dist:
            return None  # disconnected
        d = max(d, max(dist))
    return d


def closed_walks_all(n, adj, k):
    tw = [0] * (k + 1)
    for v in range(n):
        cnt = {v: 1}
        for t in range(1, k + 1):
            nxt = defaultdict(int)
            for u, w in cnt.items():
                for x in adj[u]:
                    nxt[x] += w
            cnt = nxt
            tw[t] += cnt.get(v, 0)
    return tw


def fail(msg):
    print(f"# INFEASIBLE: {msg}")
    print("Ratio: 0.0")
    sys.exit(0)


def read_ints_line(f):
    line = f.readline()
    if line == "":
        return None
    return line.split()


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path) as f:
        nH, k, n = map(int, f.readline().split())
        eps_num, eps_den = map(int, f.readline().split())
        eps2_num, eps2_den = map(int, f.readline().split())
        D_MAX, M_MAX = map(int, f.readline().split())
        mH = int(f.readline())
        H_edges = []
        adjH = defaultdict(list)
        for _ in range(mH):
            u, v = map(int, f.readline().split())
            H_edges.append((u, v))
            adjH[u].append(v)
            adjH[v].append(u)
        twH = [0] * (k + 1)
        for j in range(1, k + 1):
            twH[j] = int(f.readline())
        s2H = int(f.readline())

    B = graph_diameter(nH, adjH)
    if B is None or B <= 0:
        fail("degenerate template baseline")

    # ---- parse participant output strictly ----
    try:
        with open(out_path) as f:
            toks_line1 = read_ints_line(f)
            if toks_line1 is None or len(toks_line1) < 1:
                fail("missing N")
            N = int(toks_line1[0])
            if not (1 <= N <= n):
                fail(f"N={N} out of range [1,{n}]")
            toks_line2 = read_ints_line(f)
            if toks_line2 is None or len(toks_line2) < 1:
                fail("missing M")
            M = int(toks_line2[0])
            if not (0 <= M <= M_MAX):
                fail(f"M={M} out of range [0,{M_MAX}]")
            adjG = defaultdict(list)
            seen = set()
            for _ in range(M):
                toks = read_ints_line(f)
                if toks is None or len(toks) < 2:
                    fail("truncated edge list")
                u, v = int(toks[0]), int(toks[1])
                if not (0 <= u < N) or not (0 <= v < N) or u == v:
                    fail(f"bad edge ({u},{v})")
                key = (u, v) if u < v else (v, u)
                if key in seen:
                    fail(f"duplicate edge {key}")
                seen.add(key)
                adjG[u].append(v)
                adjG[v].append(u)
            # reject trailing garbage tokens after the declared edge count
            rest = f.read()
            if rest is not None:
                for tok in rest.split():
                    try:
                        float(tok)
                    except ValueError:
                        fail("non-numeric trailing token")
    except (ValueError, OverflowError):
        fail("malformed output (parse error)")

    # ---- finiteness / sanity of every parsed number already enforced by int()
    #      (Python's int() rejects 'nan'/'inf' outright) ----

    for v in range(N):
        if len(adjG[v]) > D_MAX:
            fail(f"vertex {v} degree {len(adjG[v])} exceeds D_MAX={D_MAX}")

    diamG = graph_diameter(N, adjG)
    if diamG is None:
        fail("G is not connected")

    twG = closed_walks_all(N, adjG, k)
    eps = Fraction(eps_num, eps_den)
    for j in range(1, k + 1):
        muG = Fraction(twG[j], N)
        muH = Fraction(twH[j], nH)
        if abs(muG - muH) > eps:
            fail(f"moment mu_{j} mismatch: G={float(muG):.6f} H={float(muH):.6f} "
                 f"tol={float(eps):.6f}")

    s2G = sum(len(adjG[v]) ** 2 for v in range(N))
    eps2 = Fraction(eps2_num, eps2_den)
    muS2G = Fraction(s2G, N)
    muS2H = Fraction(s2H, nH)
    if abs(muS2G - muS2H) > eps2:
        fail(f"degree-square-sum mismatch: G={float(muS2G):.6f} H={float(muS2H):.6f} "
             f"tol={float(eps2):.6f}")

    F = diamG
    sc = min(1000.0, 100.0 * F / max(1e-9, float(B)))
    ratio = sc / 1000.0
    if not math.isfinite(ratio):
        fail("non-finite score")
    print(f"# F(diam)={F} B(diam H)={B}")
    print("Ratio: %.6f" % ratio)
    sys.exit(0)


if __name__ == "__main__":
    main()
