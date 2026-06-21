I have a mesh of fibres and a set of traffic demands, each a (source, destination, bit-rate) triple, and I need to give every demand a route together with a piece of spectrum on each link of that route. Spectrum is the scarce resource — each fibre carries a fixed number of fine frequency slots, $12.5$ GHz wide, and the moment two demands want the same slots on the same fibre one of them is blocked. So underneath the optical-engineering vocabulary the problem is a packing problem: cram all the demands into the available spectrum without collisions, using as little spectrum as possible, where the figure of merit is the highest occupied slot index any link is forced to light up. In the older fixed-grid world this is routing-and-wavelength assignment: the spectrum is pre-carved into equal wavelength channels, each demand wants exactly one wavelength, the same wavelength must be free on every hop because there are no converters in the line, and with routes fixed the assignment is just graph colouring on the lightpath conflict graph — greedy First-Fit colouring even being optimal on a single path, since interval graphs colour perfectly. That clean picture is exactly what the elastic grid wrecks, and the failures of the obvious carry-overs are what force a new method.

The elastic grid stops carving the fibre into fat fixed channels and instead slices it into many thin slots, giving each demand *as many adjacent slots as its bit-rate needs*. A "colour" is no longer one index but a variable-width *block*, so plain colouring — which assigns a single channel per demand — simply cannot express the problem; it is the width-one, guard-free special case, not a solution to it. Interval colouring on a path captures contiguity but is welded to that one topology and ignores both the routing choice and the non-overlap of demands sharing arbitrary mesh links. Greedy spectrum-allocation rules (First-Fit, Most-Used, and the rest) are fast online policies with no optimality guarantee that react to whatever demand order and routing they are handed. And the honest exact route — a link-based multicommodity-flow integer program — is wrecked by the very thing that makes the elastic problem new: expressing "these chosen slots form one consecutive run" in linear constraints needs a thicket of auxiliary start and ordering variables, and that thicket dominates the model so badly it solves only for a handful of nodes. The need is therefore for an exact formulation that does not choke on contiguity, plus fast heuristics for scale.

I propose Routing and Spectrum Assignment built on a *channel* reformulation for the exact case and First-Fit-Decreasing (with a Most-Used variant) for the heuristic. Start with how wide a demand is, because width is where routing and spectrum couple. A transponder picks a modulation format that trades spectral efficiency against reach: $16$-QAM packs four bits per symbol into few slots but only reaches short distances, while BPSK reaches furthest and spends the most slots. The slot count is $$n = \left\lceil \frac{\text{rate}}{\text{efficiency} \times \text{slot\_BW}} \right\rceil,$$ and the admissible efficiency is fixed by whether the format's reach covers the route length. I do not re-derive that reach — whether a format survives the amplifier noise and Kerr nonlinear interference of a given path is a physical-layer noise computation upstream of allocation, handed to me as a table format $\to$ (efficiency, reach). I take the most efficient format whose reach clears the path and read off $n$, which keeps each demand as narrow as the physics allows. The coupling is already visible: a longer route forces a more robust, slot-hungrier modulation, so the *width* of a demand depends on the *route* chosen for it.

Each demand is then a block that must obey three rules — continuity (the same slots on every route link, since there is still no spectrum conversion), non-overlap (demands sharing a fibre take disjoint slots), and the genuinely new one, contiguity (the $n$ slots must be a single consecutive interval, because one transponder emits one continuous band). Contiguity is what hurts: in plain colouring any free colour works, but now I need $n$ free colours that are adjacent *and* in the same place on every link, so a link with slots free at $0,2,4,6,8$ has ample capacity yet cannot host a width-two demand. That is fragmentation, the symptom contiguity creates and colouring never had. The switch filters are not brick walls either, so adjacent blocks need a guard band of $G$ empty slots between them. The clean way to absorb the guard is to have every demand reserve $n + G$ slots — its $n$ traffic slots plus a $G$-slot moat on one side — and then forbid only overlap of the reserved blocks; any two neighbours are then automatically separated by $G$ empties and no separate guard constraint is ever written.

How hard is this? Slide the slot axis sideways and call it time: a demand becomes a task needing a contiguous time-block of length $n$, the "machines" it must run on simultaneously are the links of its route (continuity reappearing as "same time on every required machine"), and non-overlap is "a machine runs one task at a time." That is the fixed-machine scheduling problem $P|\text{fix}_j|C_{\max}$, minimising the makespan $C_{\max}$ — the highest slot index any fibre is driven to — and its three-machine version with general processing times is strongly NP-hard. So spectrum assignment is NP-hard even with routes fixed (on a ring of three links or any path of four-plus links; polynomial only on paths of at most three links), and since width-one guard-free RSA is just RWA, itself NP-hard, the elastic problem is sandwiched between two hard problems. There is no slick exact algorithm; the right targets are an exact integer program for small planning instances and fast heuristics at scale.

The decisive move for the exact program is to make contiguity true by construction instead of enforcing it. The block I assign is always a consecutive run of $w = n + G$ slots, so I pre-enumerate the runs: for width $w$ the only possible blocks are $[0, w-1], [1, w], [2, w+1], \dots$, one per start index. Call each such run a *channel*. If a demand picks exactly one channel, contiguity is automatic — a channel is consecutive by definition — and the entire thicket of consecutiveness constraints disappears. Attach the channel to a path and continuity is automatic too, because that channel is by construction the same slot interval on every link of the path. The decision variable stops being "which slots" and becomes "which (path, channel) pair," of which there are only about $W$ per width. What remains is the one easy linear family: for every link $e$ and slot $s$, at most one demand may pick a (path, channel) that covers $e$ and includes $s$. Concretely, with binary $x_{d,p,c}$ for demand $d$ on path $p$ using channel $c$, the model is $$\sum_{p,c} x_{d,p,c} = 1 \quad \forall d, \qquad \sum_{(d,p,c)\,\text{covering}\,(e,s)} x_{d,p,c} \le 1 \quad \forall e, s.$$ The guard is already inside the channel width, so non-overlap of the padded blocks *is* the guard constraint. For the objective I minimise the spectrum the network must light up, which is the highest occupied slot index — exactly the makespan $C_{\max}$, and the cost driver because the busiest fibre physically has to support that width (total slots is a legitimate alternative, but max-index is the one that maps to "how wide must the spectrum be"). A channel starting at $\text{start}$ has last occupied slot $\text{start} + n + G - 1$, so I introduce $Z$, force $$Z \ge \text{start} + n + G \quad \text{for every chosen channel}$$ (the $+1$ over the top index turns it into a slot count), and minimise $Z$. The model goes from intractable-past-six-nodes to compact. I route on $k$ shortest paths rather than letting the ILP route freely, because free multicommodity routing is what ballooned the link-based program; a few precomputed candidate paths keep continuity implicit and the variable count small, at the cost of possibly missing an exotic route — the right trade for planning instances.

For scale, when even the channel ILP is too big, I decompose: route the demand (try its $k$ shortest paths in order), then run a spectrum-allocation rule to drop a block. The rule, in the spirit of First-Fit colouring, scans start indices from zero and takes the lowest start where $[\text{start}, \text{start}+n+G-1]$ is free on every route link. Packing low needs no global state and crushes all demands toward index zero, leaving a long contiguous free tail for the next fat demand — and leaving contiguous room is precisely what fragmentation threatens. The order in which demands are fed matters enormously, and the packing structure dictates it: a fat demand needs a wide contiguous gap, so if it is placed last, after thin demands have peppered the spectrum, it can block even though it would have fit trivially into the empty grid. That is the bin-packing lesson — First-Fit-*Decreasing*: sort by width, widest first, place the hard ones while contiguous room still exists, let the thin ones trickle into the gaps. I keep a second rule too, Most-Used, which uses global knowledge of which slot indices are busy across the whole network and, among feasible starts, prefers the index already occupied on the most fibres elsewhere (ties broken to the lowest index). Piling onto an already-busy index concentrates usage and keeps *other* indices clean network-wide — and clean indices are the long contiguous gaps future fat demands need; it is the spatial-reuse counterpart of First-Fit's spectral packing, at the cost of needing global state.

The coupling of routing and spectrum is real, not bookkeeping, and it is what the decomposition can get wrong while the joint ILP gets it right. Take demands whose short routes all funnel through one shared corridor: routed on their shortest paths they pile onto the shared links, must take disjoint blocks there, and the max index climbs as each stacks above the last. But some have a longer detour over links nobody else uses; detour them and the modulation gets a little more robust and costs a few more slots, yet now they share no link with anyone and every one can start at slot zero, collapsing the max index to just the widest single demand. A route-first heuristic that always grabs the shortest path never sees this trade; the joint ILP, free to spend slots on a detour to win disjointness and reuse, finds it. That is the whole reason routing and spectrum belong in one optimisation, and why the decomposition, fast as it is, leaves something on the table. On the four-node mesh below, the route-first heuristic stacks all three demands through the short corridor 1-2-3-4 and reaches max occupied index $17$, while the exact channel ILP detours the two smaller demands onto the unused $(1,3)$ and $(2,4)$ links so all three reuse slots from index $0$, cutting the max occupied index to $8$.

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
        for i, o in enumerate(cand[d]):                      # Z >= top index + 1 = slot count
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
