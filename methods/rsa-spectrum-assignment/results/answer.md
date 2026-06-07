# Routing and Spectrum Assignment (RSA) in elastic optical networks

## Problem

A flex-grid optical network slices each fibre's spectrum into fine 12.5 GHz frequency slots. Each traffic demand (source, destination, bit-rate) must be given a route and a block of slots that satisfies four constraints: **contiguity** — the slots are a single consecutive interval; **continuity** — the identical interval is used on every link of the route (no spectrum conversion in the line); **non-overlap** — demands sharing a link occupy disjoint intervals; and a **guard band** of `G` empty slots between adjacent intervals on a link. The width of a demand depends on its bit-rate and its modulation format, which is in turn limited by route length. The offline objective is to route and assign all demands so as to minimise the spectrum used — the maximum occupied slot index over all links. The problem is NP-hard: spectrum assignment with fixed routes is the fixed-machine multiprocessor scheduling problem `P|fix_j|C_max` (NP-hard on a ring of three links and on any path of four-plus links), and the fixed-grid routing-and-wavelength-assignment problem is the width-one special case, itself NP-hard.

## Key idea

A demand needs a contiguous run of `n = ceil(rate / (efficiency × slot_BW))` slots, where the modulation's efficiency (and hence `n`) is the most efficient format whose reach covers the route — the reach being the demand-feasibility input from the physical-layer noise model. Enforcing contiguity directly in an integer program requires a thicket of consecutiveness constraints that make it intractable past a handful of nodes. The decisive move is the **channel**: for a demand of width `w = n + G`, pre-enumerate the contiguous blocks `[0,w-1], [1,w], …` (one per start index), and let the demand pick exactly one (path, channel) pair. A channel is consecutive by construction, so contiguity is automatic; attaching it to a path makes continuity automatic too. Only non-overlap remains, and with the guard folded into the channel width, non-overlap of the padded blocks *is* the guard constraint. The result is a compact ILP minimising the maximum occupied slot index. At scale, decompose into routing then spectrum assignment: order demands widest-first (First-Fit-Decreasing) and drop each into the lowest contiguous block free on every route link (First-Fit), or onto the network's most-used slot index (Most-Used) to maximise reuse.

## Algorithm

1. **Slot count.** For each demand and candidate path, pick the most spectrally efficient modulation whose reach ≥ path length; `n = ceil(rate / (efficiency × slot_BW))`. Reserve `n + G` slots (traffic + guard).
2. **Exact (channel ILP).** Pre-compute `k` shortest paths per demand. For each (path, start) build a channel covering slots `[start, start+n+G-1]`. Binary `x[d,p,c]`. Constraints: each demand uses exactly one channel; for every link `e` and slot `s`, at most one chosen channel covers `(e,s)` (non-overlap incl. guard). Bound `Z ≥ start + n + G` for every chosen channel; minimise `Z` (= max occupied slot count, guard included).
3. **Heuristic (R then SA).** Sort demands by decreasing width. For each, try its `k` shortest paths in order; on each, First-Fit the lowest start whose block is free on all route links (or Most-Used: the feasible start whose first slot is occupied on the most fibres, ties broken low). Commit and continue; report the max occupied slot index.

The coupling of routing and spectrum matters: routing a demand onto an otherwise-unused detour costs a few more slots (longer reach → more robust modulation) but lets it reuse low slot indices, which the joint ILP exploits and the route-first heuristic can miss.

## Code

```python
from math import ceil
from itertools import product
import networkx as nx

SLOT_GHZ = 12.5            # one frequency slot
GUARD = 1                 # guard-band slots between adjacent blocks on a link
NUM_SLOTS = 40            # spectral width per link, in slots

# (name, bits/s/Hz, reach_km): higher order -> fewer slots, shorter reach.
# reach is the demand-feasibility input supplied by the physical-layer model.
MODULATIONS = [("16-QAM", 4.0, 500), ("8-QAM", 3.0, 1000),
               ("QPSK", 2.0, 2000), ("BPSK", 1.0, 4000)]


def demand_slots(rate_gbps, path_len_km):
    for name, eff, reach in MODULATIONS:          # most efficient first
        if reach >= path_len_km:
            return ceil(rate_gbps / (eff * SLOT_GHZ)), name
    raise ValueError("path too long for any modulation")


def k_shortest_paths(G, s, t, k=3):
    out = []
    for i, p in enumerate(nx.shortest_simple_paths(G, s, t, weight="km")):
        if i >= k:
            break
        out.append(p)
    return out


def path_links(path):
    return [(min(u, v), max(u, v)) for u, v in zip(path, path[1:])]


def path_len(G, path):
    return sum(G[u][v]["km"] for u, v in zip(path, path[1:]))


# ---- First-Fit-Decreasing / Most-Used heuristic -------------------------
def free_block(occ, links, start, block):
    if start + block > NUM_SLOTS:
        return False
    return all(not occ[lk][j]
               for lk in links for j in range(start, start + block))


def pick_start(occ, links, block, most_used):
    feasible = [s for s in range(NUM_SLOTS - block + 1)
                if free_block(occ, links, s, block)]
    if not feasible:
        return None
    if not most_used:
        return feasible[0]                                   # First-Fit
    use = [sum(occ[lk][s] for lk in occ) for s in range(NUM_SLOTS)]
    return max(feasible, key=lambda s: (use[s], -s))         # Most-Used


def worst_width(G, dem, k):
    s, t, rate = dem
    return max(demand_slots(rate, path_len(G, p))[0]
               for p in k_shortest_paths(G, s, t, k))


def first_fit_rsa(G, demands, k=3, most_used=False):
    occ = {(min(u, v), max(u, v)): [False] * NUM_SLOTS for u, v in G.edges()}
    order = sorted(range(len(demands)),                      # FFD: widest first
                   key=lambda d: -worst_width(G, demands[d], k))
    assign = {}
    for d in order:
        s, t, rate = demands[d]
        placed = False
        for path in k_shortest_paths(G, s, t, k):
            n, mod = demand_slots(rate, path_len(G, path))
            block, links = n + GUARD, path_links(path)
            start = pick_start(occ, links, block, most_used)
            if start is not None:
                for lk in links:
                    for j in range(start, start + block):
                        occ[lk][j] = True
                assign[d] = (path, start, n, mod)
                placed = True
                break
        if not placed:
            assign[d] = None                                 # blocked
    max_idx = max((max((j for j in range(NUM_SLOTS) if occ[lk][j]), default=-1)
                   for lk in occ), default=-1)
    return assign, max_idx


# ---- Exact channel-based ILP --------------------------------------------
def build_channels(G, demands, k):
    cand = []
    for s, t, rate in demands:
        opts = []
        for path in k_shortest_paths(G, s, t, k):
            n, mod = demand_slots(rate, path_len(G, path))
            block, links = n + GUARD, frozenset(path_links(path))
            for start in range(NUM_SLOTS - block + 1):       # one channel/start
                slots = frozenset(range(start, start + block))
                top = start + block - 1                       # last occupied slot
                opts.append((path, links, start, n, mod, slots, top))
        cand.append(opts)
    return cand


def ilp_rsa(G, demands, k=3):
    cand = build_channels(G, demands, k)
    try:
        import pulp
    except ImportError:
        return _brute(cand, demands)
    prob = pulp.LpProblem("RSA", pulp.LpMinimize)
    x = {(d, i): pulp.LpVariable(f"x_{d}_{i}", cat="Binary")
         for d in range(len(demands)) for i in range(len(cand[d]))}
    Z = pulp.LpVariable("Z", lowBound=0)
    prob += Z
    for d in range(len(demands)):                            # one channel/demand
        prob += pulp.lpSum(x[(d, i)] for i in range(len(cand[d]))) == 1
        for i, o in enumerate(cand[d]):                      # Z >= top occupied slot
            prob += Z >= (o[6] + 1) * x[(d, i)]
    links = {(min(u, v), max(u, v)) for u, v in G.edges()}
    for e in links:                                          # non-overlap (+guard)
        for sslot in range(NUM_SLOTS):
            terms = [x[(d, i)] for d in range(len(demands))
                     for i, o in enumerate(cand[d])
                     if e in o[1] and sslot in o[5]]
            if terms:
                prob += pulp.lpSum(terms) <= 1
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    assign, mx = {}, -1
    for d in range(len(demands)):
        for i, o in enumerate(cand[d]):
            if pulp.value(x[(d, i)]) > 0.5:
                assign[d] = (o[0], o[2], o[3], o[4])
                mx = max(mx, o[6])
    return assign, mx


def _brute(cand, demands):                                   # verified fallback
    best = None
    for combo in product(*[range(len(c)) for c in cand]):
        used, ok = {}, True
        for d, i in enumerate(combo):
            for e in cand[d][i][1]:
                seen = used.setdefault(e, set())
                if seen & cand[d][i][5]:
                    ok = False
                    break
                seen |= cand[d][i][5]
            if not ok:
                break
        if not ok:
            continue
        cost = max(cand[d][i][6] for d, i in enumerate(combo))
        if best is None or cost < best[0]:
            best = (cost, combo)
    if best is None:
        return {}, -1
    assign = {d: (cand[d][i][0], cand[d][i][2], cand[d][i][3], cand[d][i][4])
              for d, i in enumerate(best[1])}
    return assign, best[0]


def report(assign, max_idx):
    for d in sorted(assign):
        v = assign[d]
        if v is None:
            print(f"  demand {d}: BLOCKED")
        else:
            path, start, n, mod = v
            print(f"  demand {d}: path {path} slots [{start},{start+n-1}] "
                  f"({n} sl, {mod})")
    print(f"  max occupied slot index = {max_idx}")


if __name__ == "__main__":
    G = nx.Graph()
    for u, v, km in [(1, 2, 400), (2, 3, 400), (3, 4, 400),
                     (1, 3, 1200), (2, 4, 1200)]:
        G.add_edge(u, v, km=km)
    demands = [(1, 4, 200), (1, 3, 150), (2, 4, 100)]
    print("== First-Fit-Decreasing ==");  report(*first_fit_rsa(G, demands))
    print("== Most-Used ==");             report(*first_fit_rsa(G, demands, most_used=True))
    print("== Exact ILP ==");             report(*ilp_rsa(G, demands))
```

On this 4-node mesh the route-first heuristic stacks all three demands through the short corridor (1-2-3-4) and reaches max occupied index **17**; the exact channel ILP detours the two smaller demands onto the unused (1,3) and (2,4) links so all three reuse slots from index 0, cutting the max occupied index to **8** — the routing/spectrum coupling made explicit.
