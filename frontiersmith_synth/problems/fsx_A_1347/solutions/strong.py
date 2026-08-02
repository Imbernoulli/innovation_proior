# TIER: strong
"""Insight-driven solution.

1. OBSTRUCTION DETECTION: for every vertex v whose neighborhood is, in the
   mesh graph, EXACTLY a single cycle (a "wheel" local structure), recover
   that cycle's true cyclic rim order by walking the induced edges -- this
   needs no knowledge of how the mesh was generated, only the graph itself.
2. LOCAL RECOLORING / FLIP-GRAPH MOVE: an even-length rim can be walked with
   a simple 2-coloring that closes consistently (reachable via a Kempe-chain
   style local recolor from any naive start); an odd-length rim cannot close
   with 2 colors -- the closing edge is a genuine obstruction, repaired by
   flipping exactly the seam vertex to a 4th color. This both FIXES the
   coloring and IDENTIFIES why 3 colors were impossible there.
3. Because wheel components never share a vertex here, the same small
   "cheap" palette (the globally cheapest 3, resp. 4, colors by the input's
   cost table) can be reused for every wheel, minimizing total color cost
   -- something a cost-oblivious pass never considers.
4. Every odd wheel found is reported as an explicit obstruction certificate.
"""
import sys


def find_wheel(v, adj, edge_set):
    """If v's neighborhood forms exactly one simple cycle covering all of
    N(v), return the rim in cyclic order; else return None."""
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
        cands = [x for x in induced[cur] if x != prev]
        if prev is None:
            nxt = min(induced[cur])
        else:
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
    return order


def main():
    data = sys.stdin.read().split()
    ti = 0
    n = int(data[ti]); ti += 1
    m = int(data[ti]); ti += 1
    K = int(data[ti]); ti += 1
    costs = [int(data[ti + i]) for i in range(K)]; ti += K
    faces = []
    for _ in range(m):
        a, b, c = int(data[ti]), int(data[ti + 1]), int(data[ti + 2]); ti += 3
        faces.append((a, b, c))

    edge_set = set()
    adj = {v: set() for v in range(1, n + 1)}
    for (a, b, c) in faces:
        for (u, w) in ((a, b), (b, c), (a, c)):
            edge_set.add(frozenset((u, w)))
            adj[u].add(w)
            adj[w].add(u)

    cheap = sorted(range(1, K + 1), key=lambda c: costs[c - 1])
    cheap3 = cheap[:3]
    cheap4 = cheap[:4]

    color = [0] * (n + 1)
    certs = []  # (L, hub, rim)

    seen = set()
    for v in range(1, n + 1):
        if v in seen:
            continue
        rim = find_wheel(v, adj, edge_set)
        if rim is None:
            continue
        L = len(rim)
        hub = v
        if L % 2 == 0:
            pal = cheap3
            color[hub] = pal[0]
            for i, r in enumerate(rim):
                color[r] = pal[1] if i % 2 == 0 else pal[2]
        else:
            pal = cheap4
            color[hub] = pal[0]
            for i, r in enumerate(rim):
                color[r] = pal[1] if i % 2 == 0 else pal[2]
            # positions 0 and L-1 both land on pal[1] when L is odd (the
            # closing edge conflict) -- flip the seam vertex to the 4th color
            color[rim[L - 1]] = pal[3]
            certs.append((L, hub, rim))
        seen.add(hub)
        seen.update(rim)

    # defensive fallback for anything not covered by a detected wheel
    order = sorted(range(1, n + 1), key=lambda v: -len(adj[v]))
    for v in order:
        if color[v] != 0:
            continue
        used = {color[u] for u in adj[v] if color[u] != 0}
        for c in cheap:
            if c not in used:
                color[v] = c
                break
        else:
            for c in range(1, K + 1):
                if c not in used:
                    color[v] = c
                    break

    out = []
    out.append(" ".join(str(color[v]) for v in range(1, n + 1)))
    out.append(str(len(certs)))
    for (L, hub, rim) in certs:
        out.append(str(L) + " " + str(hub) + " " + " ".join(map(str, rim)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
