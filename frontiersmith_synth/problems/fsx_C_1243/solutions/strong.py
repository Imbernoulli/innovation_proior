# TIER: strong
import sys
from collections import deque, defaultdict


def bfs_path(n, adj, src, dst):
    if src == dst:
        return [src]
    parent = [-2] * n
    parent[src] = -1
    dq = deque([src])
    while dq:
        u = dq.popleft()
        if u == dst:
            break
        for v in adj[u]:
            if parent[v] == -2:
                parent[v] = u
                dq.append(v)
    path = [dst]
    cur = dst
    while parent[cur] != -1:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path


def all_pairs_dist(n, adj):
    dist = [None] * n
    for s in range(n):
        d = [-1] * n
        d[s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if d[v] == -1:
                    d[v] = d[u] + 1
                    dq.append(v)
        dist[s] = d
    return dist


def compute_cancel_partner(gates):
    m = len(gates)
    last_touch = {}
    open_pending = {}
    partner = {}
    for i in range(1, m + 1):
        t, a, b = gates[i - 1]
        pair = (a, b) if a < b else (b, a)
        key = (pair, t)
        j = open_pending.get(key)
        if j is not None and last_touch.get(a, -1) <= j and last_touch.get(b, -1) <= j:
            partner[i] = j
            partner[j] = i
            del open_pending[key]
        else:
            open_pending[key] = i
            last_touch[a] = i
            last_touch[b] = i
    return partner


def build_initial_mapping(n, adj, edges, gates, dist):
    """The insight: place the WHOLE weighted interaction graph onto the
    coupling graph before routing a single gate, instead of leaving the
    identity layout and fixing things up gate by gate."""
    w = defaultdict(int)
    for (_t, a, b) in gates:
        key = (a, b) if a < b else (b, a)
        w[key] += 1

    if w:
        seed_pair = max(w.keys(), key=lambda k: (w[k], -k[0], -k[1]))
        a0, b0 = seed_pair
    else:
        a0, b0 = 0, (1 if n > 1 else 0)

    edges_sorted = sorted(edges)
    p0, q0 = edges_sorted[0] if edges_sorted else (0, min(1, n - 1))

    pos = {}
    used_phys = set()
    pos[a0] = p0
    used_phys.add(p0)
    if b0 != a0 and b0 not in pos:
        pos[b0] = q0
        used_phys.add(q0)

    # incidence lists for fast weight-to-placed lookups
    neigh_w = defaultdict(dict)
    for (x, y), ww in w.items():
        neigh_w[x][y] = ww
        neigh_w[y][x] = ww

    placed = set(pos.keys())
    remaining = [i for i in range(n) if i not in placed]

    while remaining:
        best_x, best_wsum = None, -1
        for x in remaining:
            s = 0
            nb = neigh_w.get(x)
            if nb:
                for y in placed:
                    s += nb.get(y, 0)
            if s > best_wsum or (s == best_wsum and (best_x is None or x < best_x)):
                best_wsum = s
                best_x = x
        x = best_x

        best_p, best_cost = None, None
        nb = neigh_w.get(x, {})
        for p in range(n):
            if p in used_phys:
                continue
            cost = 0
            for y, ww in nb.items():
                if y in pos:
                    cost += ww * dist[p][pos[y]]
            if best_cost is None or cost < best_cost or (cost == best_cost and p < best_p):
                best_cost = cost
                best_p = p
        pos[x] = best_p
        used_phys.add(best_p)
        placed.add(x)
        remaining.remove(x)

    return [pos[i] for i in range(n)]


def main():
    data = sys.stdin.read().split()
    it = iter(data)

    def nxt():
        return int(next(it))

    n = nxt(); m = nxt(); e = nxt()
    adj = [set() for _ in range(n)]
    edges = []
    for _ in range(e):
        u = nxt(); v = nxt()
        adj[u].add(v); adj[v].add(u)
        edges.append((u, v))
    gates = []
    for _ in range(m):
        t = nxt(); a = nxt(); b = nxt()
        gates.append((t, a, b))

    partner = compute_cancel_partner(gates)
    dist = all_pairs_dist(n, adj)
    mapping = build_initial_mapping(n, adj, edges, gates, dist)

    pos = mapping[:]
    at = [0] * n
    for logical, phys in enumerate(pos):
        at[phys] = logical

    ops = []

    def do_swap(u, v):
        lu, lv = at[u], at[v]
        at[u], at[v] = lv, lu
        pos[lu], pos[lv] = v, u
        ops.append("S %d %d" % (u, v))

    skip_done = set()
    for i in range(1, m + 1):
        if i in skip_done:
            continue
        j = partner.get(i)
        if j is not None and j > i:
            # both members of this cancelling pair are omitted entirely
            skip_done.add(j)
            continue
        t, a, b = gates[i - 1]
        p, q = pos[a], pos[b]
        if q not in adj[p]:
            path = bfs_path(n, adj, p, q)
            for k in range(len(path) - 2):
                do_swap(path[k], path[k + 1])
        ops.append("G %d" % i)

    out = [str(n), " ".join(map(str, mapping)), str(len(ops))]
    out.extend(ops)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
