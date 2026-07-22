# TIER: strong
# INSIGHT: the min-energy plan is NOT independent routing.  Every courier's
# shortest path shares one directed corridor (the trunk).  Detour/wait so the
# sharers cross each trunk edge on the SAME tick (form a convoy -> drafters pay
# BETA*base), and ROTATE the lead each edge so no courier accumulates a lead
# streak (the streak penalty is what makes a fixed-leader convoy pay MORE than
# riding solo).  Rotation is the whole game.
import sys, heapq

LEADER_MODE = "rotate"      # strong


def main():
    dat = sys.stdin.read().split()
    p = 0
    N = int(dat[p]); p += 1
    M = int(dat[p]); p += 1
    K = int(dat[p]); p += 1
    p += 2  # alpha beta (schedule is insensitive to their exact values)
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u = int(dat[p]); v = int(dat[p + 1]); w = int(dat[p + 2]); p += 3
        adj[u].append((v, w)); adj[v].append((u, w))
    cour = []
    for _ in range(K):
        s = int(dat[p]); g = int(dat[p + 1]); d = int(dat[p + 2]); p += 3
        cour.append((s, g, d))

    def shortest_path(src, dst):
        dist = [None] * N
        par = [-1] * N
        dist[src] = 0
        pq = [(0, src)]
        while pq:
            dd, u = heapq.heappop(pq)
            if dd > dist[u]:
                continue
            for (v, w) in adj[u]:
                nd = dd + w
                if dist[v] is None or nd < dist[v]:
                    dist[v] = nd; par[v] = u
                    heapq.heappush(pq, (nd, v))
        path = []
        cur = dst
        while cur != -1:
            path.append(cur); cur = par[cur]
        path.reverse()
        return path

    paths = [shortest_path(s, g) for (s, g, d) in cour]

    # ---- discover the shared directed corridor (edges used by >=2 couriers) ----
    use = {}
    for path in paths:
        for h in range(len(path) - 1):
            e = (path[h], path[h + 1])
            use[e] = use.get(e, 0) + 1
    shared = set(e for e, c in use.items() if c >= 2)
    if not shared:
        # no convoy opportunity -> fall back to independent ASAP routing
        return emit_solo(cour, paths)

    heads = set(v for (u, v) in shared)
    tails = set(u for (u, v) in shared)
    starts = [u for u in tails if u not in heads]
    if len(starts) != 1:
        return emit_solo(cour, paths)
    nxt = {}
    for (u, v) in shared:
        nxt.setdefault(u, []).append(v)
    order = [starts[0]]
    while order[-1] in nxt:
        vs = nxt[order[-1]]
        if len(vs) != 1:
            return emit_solo(cour, paths)
        order.append(vs[0])
    posidx = {node: k for k, node in enumerate(order)}    # trunk node -> index
    Ltrunk = len(order) - 1

    # ---- per courier: entry/exit index on the trunk + hop where it enters ----
    a_i = [0] * K; b_i = [0] * K; entry_hop = [0] * K; exit_hop = [0] * K
    on_trunk = [False] * K
    for i, path in enumerate(paths):
        idxs = [(h, posidx[node]) for h, node in enumerate(path) if node in posidx]
        if not idxs:
            on_trunk[i] = False
            continue
        on_trunk[i] = True
        entry_hop[i], a_i[i] = idxs[0]
        exit_hop[i], b_i[i] = idxs[-1]

    OFF = 0
    for i in range(K):
        if on_trunk[i]:
            OFF = max(OFF, entry_hop[i] - a_i[i])

    # ---- who leads each trunk edge k (k = trunk edge index) ----
    present = [[] for _ in range(Ltrunk)]
    for i in range(K):
        if on_trunk[i]:
            for k in range(a_i[i], b_i[i]):
                present[k].append(i)
    leader = [None] * Ltrunk
    for k in range(Ltrunk):
        pr = sorted(present[k])
        if pr:
            if LEADER_MODE == "rotate":
                leader[k] = pr[k % len(pr)]
            else:
                leader[k] = pr[0]

    # ---- build each route ----
    lines = []
    for i in range(K):
        path = paths[i]
        toks = [str(i)]
        steps = []
        if not on_trunk[i]:
            for h in range(len(path) - 1):
                steps.append(("L", path[h + 1]))
        else:
            eh = entry_hop[i]; xh = exit_hop[i]
            entry_node = order[a_i[i]]
            for h in range(eh):                         # entry spur (solo)
                steps.append(("L", path[h + 1]))
            waitc = (OFF + a_i[i]) - eh                  # wait to join the convoy
            for _ in range(max(0, waitc)):
                steps.append(("W", entry_node))
            for k in range(a_i[i], b_i[i]):             # trunk edges
                role = "L" if leader[k] == i else "D"
                steps.append((role, order[k + 1]))
            for h in range(xh, len(path) - 1):          # exit spur (solo)
                steps.append(("L", path[h + 1]))
        toks.append(str(len(steps)))
        for (act, node) in steps:
            toks.append(act); toks.append(str(node))
        lines.append(" ".join(toks))
    sys.stdout.write("\n".join(lines) + "\n")


def emit_solo(cour, paths):
    lines = []
    for i in range(len(cour)):
        path = paths[i]
        toks = [str(i), str(len(path) - 1)]
        for h in range(len(path) - 1):
            toks.append("L"); toks.append(str(path[h + 1]))
        lines.append(" ".join(toks))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
