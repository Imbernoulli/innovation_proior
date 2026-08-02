# TIER: strong
import sys
from collections import defaultdict


def lpt_slots(core_neurons, inrate, slot_cap, T):
    order = sorted(core_neurons, key=lambda x: (-inrate[x], x))
    slot_load = [0] * T
    slot_count = [0] * T
    assign = {}
    for nid in order:
        best_s = None
        for s in range(T):
            if slot_count[s] < slot_cap:
                if best_s is None or slot_load[s] < slot_load[best_s]:
                    best_s = s
        if best_s is None:
            best_s = min(range(T), key=lambda s: slot_load[s])  # overfull fallback
        assign[nid] = best_s
        slot_load[best_s] += inrate[nid]
        slot_count[best_s] += 1
    return assign


def cluster_agglomerate(N, edges, cap):
    """Capacity-capped heavy-edge agglomeration (à la METIS heavy-edge
    matching): repeatedly merge the two current clusters joined by the
    largest total spike-rate weight, as long as the merge stays within one
    core's capacity. This recovers the connectivity clusters WITHOUT ever
    looking at neuron count/fanout alone -- the insight the greedy tier
    misses entirely."""
    active = {i: True for i in range(N)}
    size = [1] * N
    parent = list(range(N))
    pairw = defaultdict(int)
    for u, v, r in edges:
        if u == v:
            continue
        key = (u, v) if u < v else (v, u)
        pairw[key] += r

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    while True:
        best_key = None
        best_w = -1
        for key, w in pairw.items():
            a, b = key
            if not active.get(a) or not active.get(b):
                continue
            if w > best_w and size[a] + size[b] <= cap:
                best_w = w
                best_key = key
        if best_key is None:
            break
        a, b = best_key
        del pairw[best_key]
        neighbors = defaultdict(int)
        to_del = []
        for key, w in list(pairw.items()):
            x, y = key
            if x == a or y == a or x == b or y == b:
                other = y if (x == a or x == b) else x
                if other == a or other == b:
                    continue
                neighbors[other] += w
                to_del.append(key)
        for key in to_del:
            del pairw[key]
        for other, w in neighbors.items():
            key = (a, other) if a < other else (other, a)
            pairw[key] += w
        active[b] = False
        size[a] += size[b]
        parent[b] = a
    members = defaultdict(list)
    for i in range(N):
        members[find(i)].append(i)
    return list(members.values())


def main():
    data = sys.stdin.read().split()
    it = iter(data)

    def nxt():
        return int(next(it))

    N = nxt(); C_max = nxt(); T = nxt(); slot_cap = nxt()
    fanout_budget = nxt(); slot_rate_budget = nxt()
    INTER = nxt(); LOCAL = nxt(); OVER = nxt()
    M = nxt()
    edges = []
    outdeg = [0] * N
    inrate = [0] * N
    for _ in range(M):
        u = nxt(); v = nxt(); r = nxt()
        edges.append((u, v, r))
        outdeg[u] += 1
        inrate[v] += r

    cap = T * slot_cap
    clusters = cluster_agglomerate(N, edges, cap)
    # Pack whole connectivity clusters (largest first) into cores, keeping
    # each cluster together even if that costs MORE cores than a blind
    # count-minimizing pack would use -- this is the mechanism the
    # innovation_hook exploits: fewer cores by dense mixing maximizes
    # inter-core spike traffic (the dominant energy term); partitioning by
    # cluster keeps that traffic local at the price of some idle capacity.
    clusters.sort(key=lambda m: -len(m))
    core_list = [[] for _ in range(C_max)]
    fanout_sum = [0] * C_max
    core_of = [-1] * N

    def place_chunk(chunk, chunk_fanout):
        for c in range(C_max):
            if len(core_list[c]) + len(chunk) <= cap and fanout_sum[c] + chunk_fanout <= fanout_budget:
                core_list[c].extend(chunk)
                fanout_sum[c] += chunk_fanout
                for j in chunk:
                    core_of[j] = c
                return True
        return False

    for group in clusters:
        gf = sum(outdeg[i] for i in group)
        if place_chunk(group, gf):
            continue
        # cluster too big for one core (rare, given generous headroom) --
        # split it into capacity-sized chunks but keep each chunk intact.
        chunk, chunk_fanout = [], 0
        for i in group:
            if len(chunk) >= cap or chunk_fanout + outdeg[i] > fanout_budget:
                place_chunk(chunk, chunk_fanout)
                chunk, chunk_fanout = [], 0
            chunk.append(i)
            chunk_fanout += outdeg[i]
        if chunk:
            place_chunk(chunk, chunk_fanout)

    # safety net (should not trigger given the generator's feasibility
    # margin): force-place any leftover neuron into the first core with any
    # room at all, ignoring fanout budget, so we never emit an out-of-range
    # sentinel core id.
    for i in range(N):
        if core_of[i] == -1:
            for c in range(C_max):
                if len(core_list[c]) < cap:
                    core_list[c].append(i)
                    core_of[i] = c
                    break
            if core_of[i] == -1:
                core_of[i] = 0
                core_list[0].append(i)

    slot_of = [0] * N
    for members in core_list:
        if not members:
            continue
        a = lpt_slots(members, inrate, slot_cap, T)
        for i, s in a.items():
            slot_of[i] = s

    out = [str(N)]
    for i in range(N):
        out.append("%d %d" % (core_of[i], slot_of[i]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
