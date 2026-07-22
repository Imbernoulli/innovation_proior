import sys, heapq


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def read_ints(path):
    return open(path).read().split()


def main():
    inp = read_ints(sys.argv[1])
    out = read_ints(sys.argv[2])

    # ---------------- parse instance ----------------
    try:
        it = iter(inp)
        N = int(next(it)); M = int(next(it)); K = int(next(it))
        alpha = float(next(it)); beta = float(next(it))
        adj = [[] for _ in range(N)]          # node -> list of (nbr, w)
        wt = {}                               # (u,v) directed -> w
        for _ in range(M):
            u = int(next(it)); v = int(next(it)); w = int(next(it))
            adj[u].append((v, w)); adj[v].append((u, w))
            wt[(u, v)] = w; wt[(v, u)] = w
        cour = []                             # (start, target, deadline)
        for _ in range(K):
            s = int(next(it)); g = int(next(it)); d = int(next(it))
            cour.append((s, g, d))
    except Exception:
        fail("bad instance")

    # ---------------- internal baseline B ----------------
    # B = sum over couriers of the min-energy (shortest-weight) route length,
    #     all solo -> exactly what an independent-shortest-path plan pays.
    def dijkstra(src, dst):
        dist = [None] * N
        dist[src] = 0
        pq = [(0, src)]
        while pq:
            dd, u = heapq.heappop(pq)
            if dist[u] is not None and dd > dist[u]:
                continue
            if u == dst:
                return dd
            for (v, w) in adj[u]:
                nd = dd + w
                if dist[v] is None or nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist[dst]

    B = 0.0
    for (s, g, d) in cour:
        sp = dijkstra(s, g)
        if sp is None:
            fail("disconnected instance")
        B += sp
    B = max(1e-9, B)

    # ---------------- parse participant output ----------------
    # K route blocks (order-free by declared index):
    #   idx  T  (act node){T}     act in {W,L,D}
    routes = [None] * K            # idx -> list of (act, node)
    try:
        p = 0
        for _ in range(K):
            idx = int(out[p]); p += 1
            T = int(out[p]); p += 1
            if idx < 0 or idx >= K:
                fail("bad courier index")
            if routes[idx] is not None:
                fail("duplicate courier %d" % idx)
            if T < 0 or T > cour[idx][2]:
                fail("route length %d exceeds deadline" % T)
            steps = []
            for _ in range(T):
                act = out[p]; p += 1
                node = int(out[p]); p += 1
                if act not in ("W", "L", "D"):
                    fail("bad action")
                if node < 0 or node >= N:
                    fail("node out of range")
                steps.append((act, node))
            routes[idx] = steps
    except SystemExit:
        raise
    except Exception:
        fail("parse error")
    if any(r is None for r in routes):
        fail("missing courier")

    # ---------------- feasibility: rebuild position timelines ----------------
    # pos[i] = list of occupied nodes over ticks 0..T_i
    pos = []
    for i in range(K):
        s, g, d = cour[i]
        cur = s
        seq = [cur]
        for (act, node) in routes[i]:
            if act == "W":
                if node != cur:
                    fail("illegal wait (courier %d)" % i)
            else:  # L or D : traverse an existing edge
                if wt.get((cur, node)) is None:
                    fail("illegal move %d->%d (courier %d)" % (cur, node, i))
                cur = node
            seq.append(cur)
        if cur != g:
            fail("courier %d does not end at target" % i)
        pos.append(seq)

    # ---------------- exact replayed energy F ----------------
    maxS = max((len(routes[i]) for i in range(K)), default=0)
    streak = [0] * K
    F = 0.0
    for s in range(maxS):
        # group active traversals at step s by directed edge
        groups = {}
        role = [None] * K          # per-courier role this step: None / 'W' / 'L' / 'D'
        for i in range(K):
            if s < len(routes[i]):
                act, node = routes[i][s]
                if act == "W":
                    role[i] = "W"
                else:
                    frm = pos[i][s]
                    e = (frm, node)
                    groups.setdefault(e, []).append((i, act))
                    role[i] = act
        # settle streaks: a courier is a "penalized leader" only inside a
        # convoy (group size >= 2) while declared 'L'.
        penleader = [False] * K
        for e, mem in groups.items():
            if len(mem) >= 2:
                for (i, act) in mem:
                    if act == "L":
                        penleader[i] = True
        for i in range(K):
            if penleader[i]:
                streak[i] += 1
            else:
                streak[i] = 0
        # charge energy
        for e, mem in groups.items():
            w = wt[e]
            if len(mem) == 1:
                F += w                                   # solo, base cost
            else:
                has_leader = any(act == "L" for (_, act) in mem)
                for (i, act) in mem:
                    if act == "L":
                        F += w * (1.0 + alpha * streak[i])
                    else:                                # drafter
                        F += w * (beta if has_leader else 1.0)

    F = max(1e-9, F)
    sc = min(1000.0, 100.0 * B / F)
    print("F=%.4f B=%.4f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
