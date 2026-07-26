# TIER: strong
# The clerk who notices the redaction cannot hide the fact that her clash
# roster came from real calendar dates. She reconstructs that hidden
# structure and uses it to decide, optimally with respect to her
# reconstruction, whom to seat.
#
# 1) HIDDEN-CHORDALITY-DETECTION. Maximum Cardinality Search (MCS) on the
#    public clash graph yields a perfect elimination ordering (valid for any
#    chordal graph, and every roster in this family secretly is one). From
#    the PEO we extract the maximal cliques, then take a MAXIMUM-WEIGHT
#    spanning forest of the clique-intersection graph -- a linear clique
#    order (its BFS-depth from a diameter endpoint) recovers the guests'
#    real calendar order, one connected roster component at a time.
# 2) SPILL-PLACEMENT-OPTIMIZATION. With a reconstructed calendar axis, "whom
#    to turn away under a `rooms` capacity" becomes weighted interval
#    scheduling with a resource limit -- solved EXACTLY on the reconstructed
#    axis via a min-cost flow (successive shortest augmenting paths): each
#    unit of flow is one room-track threading through time, an interval edge
#    of capacity 1 and cost -value lets a guest claim a track for her stay,
#    and the flow's negative-cost edges are exactly the guests worth
#    seating. This is what lets a single guest who merely LOOKS valuable,
#    but blocks a room for a long stretch, get correctly dropped in favour
#    of many smaller guests who together outvalue her.
# Room numbers are then assigned to the flow-selected guests by a sweep in
# reconstructed order, using the REAL public graph for the final feasibility
# check; anyone the sweep still can't seat (reconstruction is a heuristic,
# not guaranteed exact) is turned away on the spot, so the output is always
# valid even when the reconstruction is imperfect.
import sys, json


def solve(n, R, edges, value):
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    # ---- connected components of the clash graph ----
    comp_id = [-1] * n
    comps = []
    for s in range(n):
        if comp_id[s] != -1:
            continue
        cid = len(comps)
        stack = [s]
        comp_id[s] = cid
        members = []
        while stack:
            u = stack.pop()
            members.append(u)
            for w in adj[u]:
                if comp_id[w] == -1:
                    comp_id[w] = cid
                    stack.append(w)
        comps.append(members)

    lo = [0] * n
    hi = [0] * n
    offset = 0

    for members in comps:
        m = len(members)
        if m == 1:
            v = members[0]
            lo[v] = offset
            hi[v] = offset
            offset += 2
            continue

        # ---- Maximum Cardinality Search -> perfect elimination ordering ----
        weight = {v: 0 for v in members}
        remaining = set(members)
        visited = []
        while remaining:
            v = max(remaining, key=lambda x: (weight[x], -x))
            visited.append(v)
            remaining.discard(v)
            for u in adj[v]:
                if u in remaining:
                    weight[u] += 1
        peo = visited[::-1]
        pos = {v: i for i, v in enumerate(peo)}

        # ---- maximal cliques from the PEO ----
        cands = []
        for v in peo:
            later = frozenset(u for u in adj[v] if pos[u] > pos[v])
            cands.append(later | {v})
        cands.sort(key=lambda s: -len(s))
        cliques = []
        for s in cands:
            if not any(s <= t for t in cliques):
                cliques.append(s)
        if not cliques:
            cliques = [frozenset(members)]
        mC = len(cliques)

        # ---- maximum-weight spanning forest of the clique-intersection graph ----
        cedges = []
        for a in range(mC):
            for b in range(a + 1, mC):
                w = len(cliques[a] & cliques[b])
                if w > 0:
                    cedges.append((w, a, b))
        cedges.sort(key=lambda t: -t[0])
        parent = list(range(mC))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        ctree = [[] for _ in range(mC)]
        for w, a, b in cedges:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
                ctree[a].append(b)
                ctree[b].append(a)

        # ---- clique order via a traversal walk of each tree, per tree ----
        # (walking from one end of the tree's diameter and numbering cliques by
        # VISIT ORDER -- not graph-distance from the root -- is what keeps
        # non-adjacent siblings of a hub clique at DISTINCT positions; when a
        # few cliques share one large common core (e.g. guests that block the
        # whole window), distance-from-root collapses every sibling onto the
        # same depth and manufactures false "overlaps" between guests who
        # never really clash.)
        c_seen = [False] * mC
        c_depth = [0] * mC
        local_offset = 0
        for c0 in range(mC):
            if c_seen[c0]:
                continue
            treeset = []
            stack = [c0]
            c_seen[c0] = True
            while stack:
                x = stack.pop()
                treeset.append(x)
                for y in ctree[x]:
                    if not c_seen[y]:
                        c_seen[y] = True
                        stack.append(y)

            def bfs_dist(src):
                dist = {src: 0}
                order = [src]
                qi = 0
                while qi < len(order):
                    x = order[qi]
                    qi += 1
                    for y in ctree[x]:
                        if y not in dist:
                            dist[y] = dist[x] + 1
                            order.append(y)
                return dist, order

            d0, _ = bfs_dist(c0)
            far1 = max(d0, key=lambda k: d0[k])
            _, visit_order = bfs_dist(far1)
            for i, x in enumerate(visit_order):
                c_depth[x] = local_offset + i
            local_offset += len(visit_order) + 1

        # per-vertex lo/hi over cliques containing it
        vlo = {}
        vhi = {}
        for i, cl in enumerate(cliques):
            d = offset + c_depth[i]
            for v in cl:
                if v not in vlo or d < vlo[v]:
                    vlo[v] = d
                if v not in vhi or d > vhi[v]:
                    vhi[v] = d
        for v in members:
            lo[v] = vlo.get(v, offset)
            hi[v] = vhi.get(v, offset)
        offset += local_offset + 2

    total_d = offset + 2

    # ---- min-cost flow: R room-tracks threading through reconstructed time ----
    # graph nodes 0..total_d
    NV = total_d + 1
    g = [[] for _ in range(NV)]

    def add_edge(u, v, cap, cost):
        g[u].append([v, cap, cost, len(g[v])])
        g[v].append([u, 0, -cost, len(g[u]) - 1])

    for d in range(total_d):
        add_edge(d, d + 1, R, 0)
    for v in range(n):
        add_edge(lo[v], hi[v] + 1, 1, -value[v])

    S, T = 0, total_d
    sent = 0
    INF = float("inf")
    while sent < R:
        dist = [INF] * NV
        inq = [False] * NV
        pedge = [(-1, -1)] * NV
        dist[S] = 0
        from collections import deque
        dq = deque([S])
        inq[S] = True
        while dq:
            u = dq.popleft()
            inq[u] = False
            for ei, e in enumerate(g[u]):
                v, cap, cost, rev = e
                if cap > 0 and dist[u] + cost < dist[v] - 1e-9:
                    dist[v] = dist[u] + cost
                    pedge[v] = (u, ei)
                    if not inq[v]:
                        inq[v] = True
                        dq.append(v)
        if dist[T] >= -1e-9 or pedge[T][0] == -1:
            break
        # bottleneck
        bott = R - sent
        x = T
        while x != S:
            u, ei = pedge[x]
            bott = min(bott, g[u][ei][1])
            x = u
        x = T
        while x != S:
            u, ei = pedge[x]
            g[u][ei][1] -= bott
            rev = g[u][ei][3]
            g[x][rev][1] += bott
            x = u
        sent += bott

    # ---- which guests were kept: their interval edge has flow (residual cap 0) ----
    keep = []
    for v in range(n):
        # find the forward edge we added for v: it's g[lo[v]] entry pointing to hi[v]+1
        found_cap = 1
        for e in g[lo[v]]:
            if e[0] == hi[v] + 1 and e[2] == -value[v]:
                found_cap = e[1]
                break
        if found_cap == 0:
            keep.append(v)

    keep.sort(key=lambda v: (lo[v], hi[v], v))

    room = [-1] * n
    room_members = [set() for _ in range(R)]
    for v in keep:
        placed = False
        for r in range(R):
            if not (adj[v] & room_members[r]):
                room_members[r].add(v)
                room[v] = r
                placed = True
                break
        # if not placed, guest stays turned away (room[v] == -1)

    return room


def fallback(n, R, edges, value):
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    order = sorted(range(n), key=lambda i: (-value[i], i))
    room_members = [set() for _ in range(R)]
    room = [-1] * n
    for i in order:
        for r in range(R):
            if not (adj[i] & room_members[r]):
                room_members[r].add(i)
                room[i] = r
                break
    return room


inst = json.load(sys.stdin)
n = inst["n"]
R = inst["rooms"]
edges = inst["edges"]
value = inst["value"]

try:
    room = solve(n, R, edges, value)
    if not isinstance(room, list) or len(room) != n:
        raise ValueError("bad room")
except Exception:
    room = fallback(n, R, edges, value)

print(json.dumps({"room": room}))
