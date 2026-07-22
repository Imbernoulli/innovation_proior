# TIER: strong
"""Space-time reservation dispatcher (the innovation hook).

Processes moves in (release, id) order and keeps a RESERVATION TABLE of the
time-windows already booked on every single-lane shortcut. For each move it
prices two candidate plans against the *current* table:
  (a) the graph-shortest path, entering each shortcut at the earliest instant
      that does not overlap an already-reserved window (waiting at the node
      before it if necessary);
  (b) the shortest path that avoids every shared shortcut altogether (the ring
      detour), which by construction never needs a reservation.
It commits whichever plan finishes sooner, and only THEN books that plan's
shortcut windows into the table. Because every shortcut window a committed AGV
uses is conflict-free by construction (against every earlier commitment), no
head-on ever occurs -- the fleet trades a little wait / a longer detour for
zero deadlocks, instead of hoping the corridor is contested."""
import sys, json, heapq


def dijkstra_path(n, adj, src, dst):
    dist = [None] * n; prev = [None] * n
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if dist[u] is not None and d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if dist[v] is None or nd < dist[v]:
                dist[v] = nd; prev[v] = u
                heapq.heappush(pq, (nd, v))
    if dist[dst] is None:
        return None
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def find_free_slot(intervals, t0, length):
    """Earliest t >= t0 such that [t, t+length) overlaps none of `intervals`."""
    ivs = sorted(intervals)
    t = t0
    changed = True
    while changed:
        changed = False
        for (s, e) in ivs:
            if t < e - 1e-9 and s < t + length - 1e-9:
                t = e
                changed = True
    return t


def simulate_plan(mv, path, emap, reservations):
    """Schedule `path` against a COPY of the reservation table; return
    (times, updated_local_reservations) without touching the real table."""
    t = mv["release"]; times = [t]
    local = {k: list(v) for k, v in reservations.items()}
    for i in range(len(path) - 1):
        e = emap[frozenset((path[i], path[i + 1]))]
        if e["shared"]:
            slot = find_free_slot(local.get(e["id"], []), t, e["length"])
            t = slot
            local.setdefault(e["id"], []).append((t, t + e["length"]))
            times[-1] = t
        t += e["length"]
        times.append(t)
    return times, local


def main():
    inst = json.load(sys.stdin)
    n = inst["n_nodes"]
    adj = [[] for _ in range(n)]
    adj_safe = [[] for _ in range(n)]
    emap = {}
    for e in inst["edges"]:
        adj[e["u"]].append((e["v"], e["length"]))
        adj[e["v"]].append((e["u"], e["length"]))
        if not e["shared"]:
            adj_safe[e["u"]].append((e["v"], e["length"]))
            adj_safe[e["v"]].append((e["u"], e["length"]))
        emap[frozenset((e["u"], e["v"]))] = e

    order = sorted(inst["moves"], key=lambda m: (m["release"], m["id"]))
    reservations = {}
    out_moves = []
    for mv in order:
        path_short = dijkstra_path(n, adj, mv["src"], mv["dst"])
        path_detour = dijkstra_path(n, adj_safe, mv["src"], mv["dst"])

        times_short, res_short = simulate_plan(mv, path_short, emap, reservations)
        best_path, best_times, best_res = path_short, times_short, res_short

        if path_detour is not None:
            times_detour, res_detour = simulate_plan(mv, path_detour, emap, reservations)
            if times_detour[-1] < best_times[-1] - 1e-9:
                best_path, best_times, best_res = path_detour, times_detour, res_detour

        reservations = best_res
        out_moves.append({"id": mv["id"], "path": best_path, "times": best_times})

    print(json.dumps({"moves": out_moves}))


main()
