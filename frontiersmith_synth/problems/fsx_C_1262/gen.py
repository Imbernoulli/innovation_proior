import sys, random
from collections import defaultdict

# ---- fixed cost model (also embedded, identically, in counter.py / solutions) ----
T = 4
INTER = 45      # cost per unit spike-rate routed across a core boundary
LOCAL = 2       # cost per unit spike-rate routed within a core but across slots
OVER = 2500     # fixed cost per active (non-empty) core
CMAX_MULT = 3.5
CROSS_MULT = 1.5
CROSS_RATE = (3, 9)
MARGIN = 1.3

CASES = {
    1: ("random", dict(N=24, slot_cap=4)),
    2: ("random", dict(N=40, slot_cap=4)),
    3: ("random", dict(N=56, slot_cap=4)),
    4: ("modular", dict(K=4, cs=12)),
    5: ("modular", dict(K=5, cs=12)),
    6: ("modular", dict(K=6, cs=12)),
    7: ("modular", dict(K=7, cs=12)),
    8: ("modular", dict(K=6, cs=16)),
    9: ("modular", dict(K=8, cs=15)),
    10: ("modular", dict(K=10, cs=16)),
}


def build_edges(testId):
    rng = random.Random(1_000_003 * testId + 17)
    kind, params = CASES[testId]
    edge_rate = {}
    if kind == "random":
        N = params["N"]
        slot_cap = params["slot_cap"]
        core_cap = T * slot_cap
        K_min = -(-N // core_cap)
        C_max = max(K_min + 1, round(CMAX_MULT * K_min))
        p = 0.05
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if rng.random() < p:
                    edge_rate[(i, j)] = rng.randint(1, 20)
    else:
        K, cs = params["K"], params["cs"]
        N = K * cs
        slot_cap = -(-cs // T)
        core_cap = T * slot_cap
        C_max = max(K + 1, round(CMAX_MULT * K))
        perm = list(range(N))
        rng.shuffle(perm)
        clusters = [perm[c * cs:(c + 1) * cs] for c in range(K)]
        p_intra = 0.35
        rate_lo, rate_hi = (8, 25) if testId >= 8 else (5, 20)
        for members in clusters:
            for a in members:
                for b in members:
                    if a == b:
                        continue
                    if rng.random() < p_intra:
                        edge_rate[(a, b)] = rng.randint(rate_lo, rate_hi)
        n_cross = int(round(CROSS_MULT * cs)) * K
        for _ in range(n_cross):
            c1, c2 = rng.sample(range(K), 2)
            a = rng.choice(clusters[c1])
            b = rng.choice(clusters[c2])
            edge_rate[(a, b)] = rng.randint(*CROSS_RATE)

    edges = [(u, v, r) for (u, v), r in edge_rate.items()]
    outdeg = [0] * N
    inrate = [0] * N
    for u, v, r in edges:
        outdeg[u] += 1
        inrate[v] += r
    return N, C_max, slot_cap, edges, outdeg, inrate


# ---- reference constructions used ONLY to size fanout_budget / slot_rate_budget
#      so that the trivial / greedy / strong reference solutions (and the
#      checker's own internal baseline, which is the SAME roundrobin+LPT
#      recipe as trivial) are all guaranteed feasible on the shipped instance.

def lpt_slots(core_neurons, inrate, slot_cap, Tn):
    order = sorted(core_neurons, key=lambda x: (-inrate[x], x))
    slot_load = [0] * Tn
    slot_count = [0] * Tn
    assign = {}
    for nid in order:
        best_s = None
        for s in range(Tn):
            if slot_count[s] < slot_cap:
                if best_s is None or slot_load[s] < slot_load[best_s]:
                    best_s = s
        if best_s is None:
            return None
        assign[nid] = best_s
        slot_load[best_s] += inrate[nid]
        slot_count[best_s] += 1
    return assign


def roundrobin_cores(N, C_max, Tn, slot_cap, outdeg, fanout_budget):
    core_list = [[] for _ in range(C_max)]
    fanout_sum = [0] * C_max
    core_of = [-1] * N
    for i in range(N):
        c = i % C_max
        tries = 0
        while tries < C_max and not (len(core_list[c]) < Tn * slot_cap and fanout_sum[c] + outdeg[i] <= fanout_budget):
            c = (c + 1) % C_max
            tries += 1
        if tries >= C_max:
            return None
        core_list[c].append(i)
        fanout_sum[c] += outdeg[i]
        core_of[i] = c
    return core_of, core_list


def firstfit_cores(order, N, C_max, Tn, slot_cap, outdeg, fanout_budget):
    core_list = [[] for _ in range(C_max)]
    fanout_sum = [0] * C_max
    core_of = [-1] * N
    for i in order:
        placed = False
        for c in range(C_max):
            if len(core_list[c]) < Tn * slot_cap and fanout_sum[c] + outdeg[i] <= fanout_budget:
                core_list[c].append(i)
                fanout_sum[c] += outdeg[i]
                core_of[i] = c
                placed = True
                break
        if not placed:
            return None
    return core_of, core_list


def cluster_agglomerate(N, edges, cap):
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


def strong_cores(N, C_max, Tn, slot_cap, edges, outdeg, fanout_budget):
    cap = Tn * slot_cap
    clusters = cluster_agglomerate(N, edges, cap)
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

    for members in clusters:
        cl_fanout = sum(outdeg[i] for i in members)
        if place_chunk(members, cl_fanout):
            continue
        chunk, chunk_fanout = [], 0
        for i in members:
            if len(chunk) >= cap or chunk_fanout + outdeg[i] > fanout_budget:
                if not place_chunk(chunk, chunk_fanout):
                    return None
                chunk, chunk_fanout = [], 0
            chunk.append(i)
            chunk_fanout += outdeg[i]
        if chunk and not place_chunk(chunk, chunk_fanout):
            return None
    return core_of, core_list


def derive_budgets(N, C_max, slot_cap, edges, outdeg, inrate):
    fb_guess = sum(outdeg) + 1
    rb_guess = sum(inrate) + 1

    def sim_all(fb, rb):
        out = {}
        r = roundrobin_cores(N, C_max, T, slot_cap, outdeg, fb)
        if r is None:
            return None
        out['trivial'] = r
        order2 = sorted(range(N), key=lambda x: (-outdeg[x], x))
        r2 = firstfit_cores(order2, N, C_max, T, slot_cap, outdeg, fb)
        if r2 is None:
            return None
        out['greedy'] = r2
        r3 = strong_cores(N, C_max, T, slot_cap, edges, outdeg, fb)
        if r3 is None:
            return None
        out['strong'] = (r3[0], r3[1])
        slots = {}
        for name, (core_of, core_list) in out.items():
            slot_of = [None] * N
            for members in core_list:
                if not members:
                    continue
                a = lpt_slots(members, inrate, slot_cap, T)
                if a is None:
                    return None
                for i, s in a.items():
                    slot_of[i] = s
            slots[name] = (core_of, slot_of)
        return slots

    res = sim_all(fb_guess, rb_guess)
    if res is None:
        raise RuntimeError("generator internal error: no feasible reference assignment")

    def max_fanout_used(core_of):
        s = defaultdict(int)
        for i in range(N):
            s[core_of[i]] += outdeg[i]
        return max(s.values()) if s else 0

    def max_rate_used(core_of, slot_of):
        s = defaultdict(int)
        for i in range(N):
            s[(core_of[i], slot_of[i])] += inrate[i]
        return max(s.values()) if s else 0

    max_fb = max(max_fanout_used(res[k][0]) for k in res)
    max_rb = max(max_rate_used(res[k][0], res[k][1]) for k in res)
    fanout_budget = int(max_fb * MARGIN) + 1
    slot_rate_budget = int(max_rb * MARGIN) + 1

    res2 = sim_all(fanout_budget, slot_rate_budget)
    if res2 is None:
        raise RuntimeError("generator internal error: budgets with margin still infeasible")
    return fanout_budget, slot_rate_budget


def main():
    testId = int(sys.argv[1])
    N, C_max, slot_cap, edges, outdeg, inrate = build_edges(testId)
    fanout_budget, slot_rate_budget = derive_budgets(N, C_max, slot_cap, edges, outdeg, inrate)

    lines = []
    lines.append("%d %d %d %d %d %d %d %d %d" % (
        N, C_max, T, slot_cap, fanout_budget, slot_rate_budget, INTER, LOCAL, OVER))
    lines.append(str(len(edges)))
    for u, v, r in edges:
        lines.append("%d %d %d" % (u, v, r))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
