#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the mesh-coloring-with-proof problem.

Input format (see gen.py):
    n m K
    K ints: cost[1..K]
    m lines: "a b c" (1-indexed vertex ids of a triangular face)

Participant output (whitespace-separated tokens, any layout):
    n ints: color[1..n], each in [1,K]
    1 int:  C = number of obstruction certificates (C >= 0)
    C lines, each: "L hub r_1 r_2 ... r_L"
        L        = claimed rim length (L >= 3)
        hub      = a vertex id
        r_1..r_L = the rim vertex ids in cyclic order
    A certificate is a claim "the subgraph {hub}+{r_1..r_L} is an odd wheel,
    hence this mesh needs >=4 colors" -- it is verified structurally against
    the real mesh edges and only credited if true (L odd AND every claimed
    hub-rim / rim-rim edge really exists).

Feasibility (hard failure -> Ratio: 0.0):
    - output must parse as n well-formed ints in [1,K] for the coloring
    - the coloring must be PROPER: no two vertices sharing a mesh edge
      (from any of the m faces) may share a color
Certificates are a separate, softly-scored bonus/penalty layer that can
never by itself force Ratio: 0.0 (malformed certificate lines are simply
discarded and do not affect the main coloring's validity).
"""
import sys


def fail0(reason):
    print("INVALID(%s) Ratio: 0.000000" % reason)
    sys.exit(0)


def check_cert(hub, rim, edge_set, n):
    L = len(rim)
    if L < 3 or L > n:
        return False
    if not (1 <= hub <= n):
        return False
    if any(not (1 <= r <= n) for r in rim):
        return False
    ids = [hub] + rim
    if len(set(ids)) != len(ids):
        return False
    for r in rim:
        if frozenset((hub, r)) not in edge_set:
            return False
    for i in range(L):
        a, b = rim[i], rim[(i + 1) % L]
        if frozenset((a, b)) not in edge_set:
            return False
    return True


def find_wheel(v, adj, edge_set):
    """If v's neighborhood forms exactly one simple cycle covering all of
    N(v), return that rim's length; else None. Used only to independently
    count how many genuine odd-wheel obstructions the mesh actually
    contains, so the certificate bonus can be normalized against the truth
    (bounded score contribution regardless of instance size)."""
    nbrs = list(adj[v])
    if len(nbrs) < 4:
        return None
    nbr_set = set(nbrs)
    induced = {u: set() for u in nbrs}
    for u in nbrs:
        for w in adj[u]:
            if w in nbr_set and w != u:
                induced[u].add(w)
    for u in nbrs:
        if len(induced[u]) != 2:
            return None
    start = min(nbrs)
    order = [start]
    prev, cur = None, start
    while True:
        if prev is None:
            nxt = min(induced[cur])
        else:
            cands = [x for x in induced[cur] if x != prev]
            if len(cands) != 1:
                return None
            nxt = cands[0]
        if nxt == start:
            break
        if nxt in order:
            return None
        order.append(nxt)
        prev, cur = cur, nxt
    if len(order) != len(nbrs):
        return None
    return len(order)


def count_true_odd_wheels(n, adj, edge_set):
    seen = set()
    count = 0
    for v in range(1, n + 1):
        if v in seen:
            continue
        L = find_wheel(v, adj, edge_set)
        if L is not None:
            seen.add(v)
            seen.update(adj[v])
            if L % 2 == 1:
                count += 1
    return count


def compute_F(colors, costs, K, n, valid_certs, invalid_attempts, true_odd_count):
    distinct = len(set(colors))
    total_cost = sum(costs[c - 1] for c in colors)
    max_cost = n * max(costs)
    A = 14.0            # weight: fewer distinct colors used
    Bw = 5.0            # weight: cheaper color budget spent
    CERT_BONUS_MAX = 28.0  # full marks for certifying EVERY true obstruction
    CERT_PENALTY = 5.0     # cost per bogus / duplicate / malformed certificate
    # normalized so the bonus is bounded regardless of how many wheels the
    # instance happens to contain (prevents unbounded score growth on
    # larger/denser instances -- a fixed fraction of a fixed cap, not a
    # per-item reward that scales with instance size)
    cert_frac = min(1.0, valid_certs / true_odd_count) if true_odd_count > 0 else 0.0
    F = (A * (K - distinct)
         + Bw * (max_cost - total_cost) / n
         + CERT_BONUS_MAX * cert_frac
         - CERT_PENALTY * invalid_attempts)
    return max(F, 0.0)


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path) as f:
        toks = f.read().split()
    ti = 0
    n = int(toks[ti]); ti += 1
    m = int(toks[ti]); ti += 1
    K = int(toks[ti]); ti += 1
    costs = [int(toks[ti + i]) for i in range(K)]; ti += K
    faces = []
    for _ in range(m):
        a, b, c = int(toks[ti]), int(toks[ti + 1]), int(toks[ti + 2]); ti += 3
        faces.append((a, b, c))

    edge_set = set()
    adj = {v: set() for v in range(1, n + 1)}
    for (a, b, c) in faces:
        for (u, v) in ((a, b), (b, c), (a, c)):
            edge_set.add(frozenset((u, v)))
            adj[u].add(v)
            adj[v].add(u)

    # ---- strict parse + validate the coloring artifact ----
    try:
        with open(out_path) as f:
            otoks = f.read().split()
        oi = 0
        colors = []
        for _ in range(n):
            colors.append(int(otoks[oi])); oi += 1
        for c in colors:
            if not (1 <= c <= K):
                fail0("color_out_of_range")
        for e in edge_set:
            u, v = tuple(e)
            if colors[u - 1] == colors[v - 1]:
                fail0("adjacent_same_color")
    except SystemExit:
        raise
    except Exception:
        fail0("parse_error")

    # ---- soft-parse the certificate block (never invalidates the coloring) ----
    valid_certs = 0
    invalid_attempts = 0
    try:
        C = int(otoks[oi]); oi += 1
        if not (0 <= C <= 2000):
            C = 0
        seen_hubs = set()
        for _ in range(C):
            L = int(otoks[oi]); oi += 1
            if L < 3 or L > n:
                invalid_attempts += 1
                break  # cannot safely resync token stream -> stop reading certs
            hub = int(otoks[oi]); oi += 1
            rim = [int(otoks[oi + i]) for i in range(L)]; oi += L
            ok = check_cert(hub, rim, edge_set, n) and (L % 2 == 1)
            if ok and hub not in seen_hubs:
                valid_certs += 1
                seen_hubs.add(hub)
            else:
                invalid_attempts += 1
    except Exception:
        pass

    true_odd_count = count_true_odd_wheels(n, adj, edge_set)
    F = compute_F(colors, costs, K, n, valid_certs, invalid_attempts, true_odd_count)

    # ---- checker's own internal baseline: color each connected patch with a
    # FRESH block of colors (never reusing a palette across independent
    # patches), cost-blind, no certificates ----
    def components():
        seen = [False] * (n + 1)
        comps = []
        for s in range(1, n + 1):
            if seen[s]:
                continue
            stack = [s]
            seen[s] = True
            comp = []
            while stack:
                u = stack.pop()
                comp.append(u)
                for w in adj[u]:
                    if not seen[w]:
                        seen[w] = True
                        stack.append(w)
            comp.sort()
            comps.append(comp)
        comps.sort(key=lambda c: c[0])
        return comps

    bcolor = [0] * (n + 1)
    block_start = 1
    for comp in components():
        palette = [((block_start - 1 + i) % K) + 1 for i in range(K)]
        local = {}
        for v in comp:
            used = {local[u] for u in adj[v] if u in local}
            for c in palette:
                if c not in used:
                    local[v] = c
                    break
        for v in comp:
            bcolor[v] = local[v]
        distinct_here = len(set(local.values()))
        block_start = ((block_start - 1 + distinct_here) % K) + 1
    B = compute_F(bcolor[1:], costs, K, n, 0, 0, true_odd_count)
    if B <= 1e-9:
        B = 1e-9

    sc = min(1000.0, 100.0 * F / B)
    print("F=%.4f B=%.4f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
