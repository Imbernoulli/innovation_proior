# TIER: strong
"""Eulerian/TSP hybrid on the symmetry quotient -- the insight the trap hides.

The design is an ORBIT: congruent copies of one multicolor motif placed on a
wide translation lattice. The obvious recipe locks to one color at a time GLOBALLY
(minimal color loads: C), but every color's arcs are scattered across all the
far-apart copies, so each of the C sweeps pays a full lattice crossing of long
jumps. The group structure prices the alternative: motif diameter << lattice
spacing, so sweeping the ORBIT once and re-buying the C color loads inside each
region is strictly cheaper. This solution:

  1. ORBIT (TSP) SIDE: splits the design into its connected components (the
     orbit regions) and orders them along ONE short tour (nearest neighbor +
     2-opt over region centroids), paying the long-distance jump mileage once.
  2. QUOTIENT (Eulerian) SIDE: inside each region, decomposes each color's
     subgraph into edge-disjoint trails (Hierholzer, odd-degree starts first)
     -- one trail cover solves the traversal problem for the orbit
     representative; every congruent copy reuses the same plan shape.
  3. PRICED INTERLEAVING: color groups inside a region are visited in order of
     true marginal price (dist to the group's nearest trail start + K if the
     color is not loaded), and trails inside a group by nearest-neighbor jumps
     -- so a color change is re-bought exactly when the same-color trails are
     farther than K, and the region ENTRY prefers the incoming thread color
     when the geometry allows it.

The result deliberately pays R*C color loads instead of C, converting C-1
lattice crossings into short local jumps.
"""
import sys


def main():
    data = sys.stdin.read().split()
    p = 0

    def ni():
        nonlocal p
        v = int(data[p]); p += 1
        return v

    V = ni(); E = ni(); C = ni(); K = ni()
    xs = [0] * V; ys = [0] * V
    for i in range(V):
        xs[i] = ni(); ys[i] = ni()
    edges = [(ni(), ni(), ni()) for _ in range(E)]

    def dist(a, b):
        return abs(xs[a] - xs[b]) + abs(ys[a] - ys[b])

    # ---- connected components = orbit regions ----
    adjv = [[] for _ in range(V)]
    for idx, (u, v, c) in enumerate(edges):
        adjv[u].append(v)
        adjv[v].append(u)
    comp = [-1] * V
    ncomp = 0
    for s in range(V):
        if comp[s] != -1 or not adjv[s]:
            continue
        stack = [s]
        comp[s] = ncomp
        while stack:
            x = stack.pop()
            for w in adjv[x]:
                if comp[w] == -1:
                    comp[w] = ncomp
                    stack.append(w)
        ncomp += 1

    # ---- per region, per color: Eulerian trail cover (quotient side) ----
    region_color_edges = [[[] for _ in range(C)] for _ in range(ncomp)]
    for idx, (u, v, c) in enumerate(edges):
        region_color_edges[comp[u]][c].append((u, v, idx))

    # trails[rg] = list of (color, [vertices]) -- grouped by color
    region_groups = []
    for rg in range(ncomp):
        groups = []  # (color, [trail, ...])
        for c in range(C):
            es = region_color_edges[rg][c]
            if not es:
                continue
            adj = {}
            for (u, v, idx) in es:
                adj.setdefault(u, []).append((v, idx))
                adj.setdefault(v, []).append((u, idx))
            for w in adj:
                adj[w].sort()
            used = set()
            trails = []
            starts = sorted(adj, key=lambda x: (len(adj[x]) % 2 == 0, x))
            for s in starts:
                while True:
                    if all(idx in used for (_, idx) in adj[s]):
                        break
                    seq = [s]
                    cur = s
                    while True:
                        nxt = -1; nidx = -1
                        for (w, idx) in adj[cur]:
                            if idx not in used:
                                nxt = w; nidx = idx
                                break
                        if nxt < 0:
                            break
                        used.add(nidx)
                        seq.append(nxt)
                        cur = nxt
                    trails.append(seq)
            groups.append((c, trails))
        region_groups.append(groups)

    # ---- orbit side: TSP tour over region centroids (NN + 2-opt) ----
    cent = []
    for rg in range(ncomp):
        vs = [v for c in range(C) for (u, v, idx) in region_color_edges[rg][c]
              for v in (u, v)]
        cent.append((sum(xs[v] for v in vs) / len(vs),
                     sum(ys[v] for v in vs) / len(vs)))

    def cdist(a, b):
        return abs(cent[a][0] - cent[b][0]) + abs(cent[a][1] - cent[b][1])

    start = min(range(ncomp),
                key=lambda rg: (abs(cent[rg][0] - xs[0]) + abs(cent[rg][1] - ys[0]), rg))
    tour = [start]
    left = set(range(ncomp)) - {start}
    while left:
        last = tour[-1]
        nxt = min(left, key=lambda rg: (cdist(last, rg), rg))
        tour.append(nxt)
        left.discard(nxt)
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if j + 1 >= n:
                    continue
                a = tour[i]; b = tour[i + 1]
                c = tour[j]; d = tour[j + 1]
                if cdist(a, c) + cdist(b, d) < cdist(a, b) + cdist(c, d):
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True

    # ---- execute: one orbit tour; inside a region, price color re-buys ----
    ops = []
    pos = 0
    curcol = -1
    for rg in tour:
        groups = [(c, list(trails)) for (c, trails) in region_groups[rg]]
        while groups:
            # pick the color group whose nearest trail start is cheapest,
            # priced as jump distance + (K if the color is not loaded)
            bestg = None
            for gi, (c, trails) in enumerate(groups):
                for ti, seq in enumerate(trails):
                    for end in (0, len(seq) - 1):
                        price = (dist(pos, seq[end])
                                 + (0 if c == curcol else K))
                        cand = (price, dist(pos, seq[end]), seq[end], c, gi)
                        if bestg is None or cand < bestg:
                            bestg = cand
            gi = bestg[4]
            c, trails = groups.pop(gi)
            # chain this color's trails by nearest-neighbor local jumps
            while trails:
                best = None
                for ti, seq in enumerate(trails):
                    for end in (0, len(seq) - 1):
                        d = dist(pos, seq[end])
                        cand = (d, seq[end], ti)
                        if best is None or cand < best:
                            best = cand
                d, target, ti = best
                seq = trails.pop(ti)
                if seq[0] != target:
                    seq = seq[::-1]
                if pos != seq[0]:
                    ops.append("J %d" % seq[0])
                    pos = seq[0]
                for k in range(len(seq) - 1):
                    ops.append("S %d %d" % (seq[k], seq[k + 1]))
                pos = seq[-1]
            curcol = c
    out = [str(len(ops))]
    out.extend(ops)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
