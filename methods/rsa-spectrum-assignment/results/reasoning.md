I have a mesh of fibres and a pile of traffic demands, and I need to give each demand a route and a piece of spectrum on every link of that route. The thing I keep coming back to is that spectrum is the scarce resource — each fibre has a fixed width, and once two demands want the same colour on the same fibre, one of them is blocked. So the real question is a packing question: how do I cram all these demands into the available spectrum without collisions, and how little spectrum can I get away with?

Let me start where I actually know the answer, the fixed-grid world, and then watch it break. There, the spectrum is pre-carved into equal wavelength channels, every demand wants exactly one wavelength, and because the line has no wavelength converters, a demand must hold the *same* wavelength on every hop of its route. That's the wavelength-continuity rule. Two demands on the same fibre need different wavelengths. If I fix the routes, the assignment is pure graph colouring: make a vertex per lightpath, draw an edge between any two lightpaths that share a fibre, colour the graph with as few wavelengths as possible. Greedy First-Fit colouring — give each lightpath the lowest free wavelength — is the standard move, and on a single path it's even optimal because the conflict graph is an interval graph and interval graphs colour perfectly. Good. I have a clean mental model and a clean algorithm.

Now the elastic grid wrecks it. The fibre isn't carved into fat fixed channels anymore; it's sliced into many thin frequency slots, 12.5 GHz each, and a demand is given *as many adjacent slots as its bit-rate needs*. A fat demand takes eight slots, a thin one takes two. So the "colour" of a demand is no longer a single index — it's a *block* of slots, and the block width varies per demand. That's the whole point of going elastic: stop wasting a fat fixed channel on a thin demand. But it means my colouring picture has to change, because I'm no longer assigning one colour, I'm assigning a contiguous *run* of colours whose length depends on the demand.

How many slots does a demand actually need? It carries some bit-rate, and the transponder picks a modulation format. Here's the lever: a high-order format like 16-QAM packs four bits per symbol per polarisation, so the bit-rate fits into few slots — but it needs a fat signal-to-noise margin and only reaches a short distance. Drop to QPSK, two bits per symbol, and it reaches twice as far but spends twice the slots; BPSK reaches furthest and costs the most slots. So the slot count is `n = ceil(rate / (efficiency × slot_bandwidth))`, and which efficiency I'm allowed to use depends on whether that format's reach covers the route length. I do *not* want to re-derive the reach here — whether 16-QAM survives 800 km of fibre after amplifier noise and Kerr nonlinear interference is a physical-layer computation, the analytic noise model of the link, and it hands me a table: format → efficiency, reach. I'll take the most spectrally efficient format whose reach clears the path, and read off `n`. That keeps each demand's slot count as small as the physics allows. Notice this already couples to routing: a longer route forces a more robust, slot-hungrier modulation, so the *width* of a demand depends on the *route* I pick for it. I'll come back to that — it bites later.

So now each demand is a block of `n` adjacent slots. What rules must the block obey? Two of them carry straight over and one is genuinely new. Continuity carries over: the same slots on every link of the route, because still no spectrum conversion in the line. Non-overlap carries over: two demands sharing a fibre can't occupy the same slots. The new one is contiguity — the `n` slots must be *consecutive*, a single interval, because one transponder emits one continuous band of spectrum. I can't give a demand slots 2, 5, and 9; it has to be 2, 3, 4.

Let me feel why contiguity is the thing that hurts. In plain colouring, when I need a colour, *any* free colour works — the free colours don't have to be next to each other. Now I need `n` free colours that are *adjacent*, and adjacent *in the same place* on every link of the route. Picture a link with slots free at 0, 2, 4, 6, 8 and occupied at the odd indices: it has five free slots, plenty of capacity, but it can't host a demand needing two adjacent slots, because no two free slots touch. That's fragmentation, and it's the symptom that contiguity creates and colouring never had. A link can be far from full and still block a demand. So my objective can't just be "don't run out of slots" — it's "keep contiguous room," which is subtler.

One more physical wrinkle before I formalise: the switch filters aren't brick walls. Two demands packed slot-against-slot on a fibre would bleed into each other through filter roll-off and crosstalk. So I need a guard band — a slot or two of empty space — between adjacent occupied blocks on a link. Cleanest way to bake that in: have each demand *reserve* its `n` traffic slots plus `G` guard slots, and then just forbid overlap of the reserved blocks. If every demand carries its own guard on one side, any two adjacent demands are automatically separated by `G` empties and I never have to write a separate guard constraint. So a demand of width `n` occupies a block of `n + G` slots; the traffic lives in the first `n`, the last `G` are the moat.

Now, how hard is this? Let me try to see the assignment subproblem — routes fixed — as something I already understand, so I know what I'm up against. Slide the slot axis sideways and call it time. A demand becomes a *task* that needs a contiguous time-block of length `n` (its slot count). The "machines" the task must run on are the links of its route, and it must run on all of them *simultaneously*, in the same time-block — that's exactly the continuity constraint reappearing as "same time on every required machine." Non-overlap on a fibre is "a machine does one task at a time." So spectrum assignment is scheduling tasks, each pinned to a fixed subset of machines, with no preemption, minimising the makespan — the latest finishing time, which is the highest slot index any fibre is driven to. That's the `P|fix_j|C_max` problem. And the three-machine version with free processing times is known to be strongly NP-hard. So even with routes fixed, on a ring of three links or a path of four-plus links, this is NP-hard; only on paths of at most three links is it polynomial. The colouring view says the same thing from the other side: on a single path it's the interval-chromatic-number problem with a known `(2+ε)` approximation, but the general mesh has no such easy out.

And I haven't even added routing yet. But here's a sanity check from below: if I shrink every demand to width one slot and one guard-free unit, I'm back to assigning one channel per demand with continuity — that's RWA, which is itself NP-hard. So the elastic problem is sandwiched between two hard problems; it's not going to have a slick exact algorithm. Fine. That tells me what to build: an exact integer program for small planning instances, and fast heuristics for scale.

Let me try the honest, brute-force formulation first and watch where it gets ugly, because the ugliness will tell me what trick to reach for. Index every slot `0..W-1`. For each demand introduce, say, a binary `y[d,e,s]` = demand `d` uses slot `s` on link `e`, plus a start variable. Continuity: `y[d,e,s]` is the same for all `e` on the route. Non-overlap: for each link and slot, `sum_d y[d,e,s] ≤ 1`. Contiguity: the set `{s : y[d,e,s]=1}` must be an interval of length `n_d` — and *this* is the painful one. Expressing "these chosen slots form one consecutive run" in linear constraints needs a thicket of auxiliary start/indicator variables and ordering constraints, and it's exactly this thicket that blows the ILP up so it only solves for a handful of nodes. The contiguity bookkeeping is the bottleneck. I keep writing constraints whose only job is to force consecutiveness, and they dominate the model.

So stare at contiguity and ask: what if I never have to *enforce* it, because it's true by construction? The block I'm assigning is always a contiguous run of `n_d + G` slots. What if I pre-enumerate the runs? For a demand of width `w = n_d + G`, the only blocks it could possibly take are `[0, w-1]`, `[1, w]`, `[2, w+1]`, … one per start index. Call each such block a *channel*. If I make the demand pick exactly one channel, then contiguity is automatic — a channel is a consecutive run by definition — and I've deleted the entire thicket of consecutiveness constraints. The demand's decision variable isn't "which slots" but "which channel," and there are only about `W` of them per width. Continuity also comes for free if I attach the channel to a *path*: the demand picks one (path, channel) pair, and that channel is, by construction, the same slot interval on every link of that path. So both contiguity and continuity vanish into the variable definition. What's left to constrain is only non-overlap, and that's the easy linear one: for every link `e` and every slot `s`, at most one demand may pick a (path, channel) that covers `e` and includes `s`.

That's the move. Variable `x[d,p,c]` = demand `d` routed on path `p` using channel `c`. One channel per demand: `sum_{p,c} x[d,p,c] = 1`. Non-overlap: for each link `e`, each slot `s`, `sum over (d,p,c) covering e and s of x[d,p,c] ≤ 1`. Guard is already inside the channel width `n_d + G`, so non-overlap of the padded blocks *is* the guard constraint — I don't write a separate one. And the objective: I want to minimise the spectrum the network is forced to light up, which is the highest slot index any demand occupies — and the guard slots are reserved spectrum too, so they count. A channel starting at `start` has width `n_d + G`, so its last occupied slot is `start + n_d + G - 1`. Introduce `Z`, force `Z ≥ start + n_d + G` for every chosen channel (the `+1` over the index makes it a count of slots, index plus one), and minimise `Z`. Now contiguity and continuity are free, non-overlap and guard are one clean family of constraints, and the objective is a single max. The model went from intractable-past-six-nodes to compact. That's the channel reformulation paying off.

Why minimise the *max* index rather than, say, total slots used? Because the max index on the busiest fibre is what that fibre physically has to support — it's the spectral width I have to provision, the cost driver — and it's exactly the makespan `C_max` from the scheduling view. Total-slots is a legitimate alternative objective, but max-index is the one that maps to "how wide does the spectrum have to be," so that's what I'll bound.

Why route on `k` shortest paths instead of letting the ILP route freely over the whole graph? Free multicommodity routing balloons the model again — that's what made the joint link-based ILP intractable. Pre-computing a few candidate paths per demand and letting the ILP choose among them keeps continuity implicit (a chosen path is a fixed set of links) and keeps the variable count down, at the cost of possibly missing an exotic route. For planning instances that's the right trade.

Now the heuristic, for when even the channel ILP is too big. Routes and assignment are coupled, but coupling them exactly is what's expensive, so decompose: route the demand (try its `k` shortest paths in order), then run a spectrum-allocation rule to drop a block. The rule, in the spirit of First-Fit colouring: scan start indices from zero, take the *lowest* start where the block `[start, start + n + G - 1]` is free on *every* link of the route. Lowest-index-first does two good things — it needs no global state, and it crushes all demands down toward index zero, which leaves a long contiguous free tail at the top for whatever fat demand comes next. Leaving contiguous room is exactly the thing fragmentation threatens, so packing low is the right instinct.

But the *order* I feed demands to First-Fit matters enormously, and I can reason out the right order from the packing structure. A fat demand needs a wide contiguous gap; if I place it last, after the thin demands have peppered the spectrum, there may be no wide gap left and it blocks — even though, placed first into the empty spectrum, it would have fit easily. This is the bin-packing lesson: First-Fit-*Decreasing*. Sort demands by width, widest first, and place the hard ones while contiguous room still exists; let the thin demands trickle into the gaps afterward. So the heuristic is: order by decreasing slot count, route on `k` paths, First-Fit the lowest contiguous block on all route links, commit, move on; report the max occupied index at the end.

There's a second allocation rule worth having, because First-Fit's lowest-index greed isn't the only good idea. Suppose I have global knowledge of which slot indices are busy across the *whole* network. Most-Used says: among the feasible starts for this demand, prefer the slot index that is already occupied on the most fibres elsewhere. Why would I want to pile onto an already-busy index? Because reusing the same index in many places concentrates usage onto a few indices and keeps the *other* indices completely clean across the network — and clean indices are exactly the long contiguous gaps that future fat demands need. It's the spatial-reuse counterpart of First-Fit's spectral packing: First-Fit packs each link low; Most-Used packs the network's *index usage* onto common slots. The cost is that it needs global state and is slower to evaluate; First-Fit needs neither. I'll keep both and break Most-Used ties with First-Fit (lowest index).

Let me make sure I see why the *coupling* of routing and spectrum is real and not just bookkeeping, because it's the thing the decomposed heuristic can get wrong and the joint ILP can get right. Take three demands on a small mesh where the short routes all funnel through one shared corridor of links. If I route everyone on their shortest path, they all pile onto the shared links, must take *disjoint* slot blocks there, and the max index climbs — each demand stacked above the last. But some of those demands have a longer detour route over links that *nobody else uses*. Detour them: the modulation gets more robust and they cost a few more slots each, but now they share no link with anyone, so every one of them can start at slot zero. The max index collapses to just the widest single demand. A route-first heuristic that always grabs the shortest path never sees this; the joint ILP, free to trade a longer route for spectral reuse, finds it. That trade — spend extra slots on a detour to win disjointness and reuse — is the whole reason routing and spectrum belong in one optimisation, and it's why the decomposition, fast as it is, leaves something on the table.

Time to write it. Topology in a graph with kilometre weights; `k`-shortest-paths from the library; a per-link boolean occupancy array. The slot-count rule reads the modulation table. The heuristic sorts by worst-case width, routes, First-Fits (with a Most-Used variant). The exact solver enumerates (path, channel) candidates and runs the channel ILP — one channel per demand, per-link-per-slot non-overlap, minimise the max occupied slot index — falling back to a verified exhaustive search over channel combinations if no ILP solver is around, so the logic is checkable either way.

```python
from math import ceil
from itertools import product
import networkx as nx

SLOT_GHZ = 12.5            # one frequency slot
GUARD = 1                 # guard-band slots between adjacent blocks on a link
NUM_SLOTS = 40            # spectral width per link, in slots

# (name, bits/s/Hz, reach_km): higher order -> fewer slots, shorter reach.
# reach is the demand-feasibility input from the physical-layer noise model.
MODULATIONS = [("16-QAM", 4.0, 500), ("8-QAM", 3.0, 1000),
               ("QPSK", 2.0, 2000), ("BPSK", 1.0, 4000)]

def demand_slots(rate_gbps, path_len_km):
    for name, eff, reach in MODULATIONS:          # most efficient first
        if reach >= path_len_km:                  # reach must cover the route
            return ceil(rate_gbps / (eff * SLOT_GHZ)), name
    raise ValueError("path too long for any modulation")

def k_shortest_paths(G, s, t, k=3):
    out = []
    for i, p in enumerate(nx.shortest_simple_paths(G, s, t, weight="km")):
        if i >= k: break
        out.append(p)
    return out

def path_links(path):
    return [(min(u, v), max(u, v)) for u, v in zip(path, path[1:])]
def path_len(G, path):
    return sum(G[u][v]["km"] for u, v in zip(path, path[1:]))

# --- First-Fit-Decreasing heuristic: route then assign -------------------
def free_block(occ, links, start, block):
    # continuity (same slots on all route links) + non-overlap with placed
    if start + block > NUM_SLOTS: return False
    return all(not occ[lk][j] for lk in links for j in range(start, start + block))

def pick_start(occ, links, block, most_used):
    feasible = [s for s in range(NUM_SLOTS - block + 1)
                if free_block(occ, links, s, block)]
    if not feasible: return None
    if not most_used: return feasible[0]                      # First-Fit
    use = [sum(occ[lk][s] for lk in occ) for s in range(NUM_SLOTS)]
    return max(feasible, key=lambda s: (use[s], -s))          # Most-Used

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
        s, t, rate = demands[d]; placed = False
        for path in k_shortest_paths(G, s, t, k):
            n, mod = demand_slots(rate, path_len(G, path))
            block, links = n + GUARD, path_links(path)
            start = pick_start(occ, links, block, most_used)
            if start is not None:
                for lk in links:
                    for j in range(start, start + block): occ[lk][j] = True
                assign[d] = (path, start, n, mod); placed = True; break
        if not placed: assign[d] = None                      # blocked
    max_idx = max((max((j for j in range(NUM_SLOTS) if occ[lk][j]), default=-1)
                   for lk in occ), default=-1)
    return assign, max_idx

# --- Exact channel ILP: a channel folds in contiguity+continuity ---------
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
    Z = pulp.LpVariable("Z", lowBound=0); prob += Z
    for d in range(len(demands)):                            # one channel/demand
        prob += pulp.lpSum(x[(d, i)] for i in range(len(cand[d]))) == 1
        for i, o in enumerate(cand[d]):                      # Z >= top occupied slot
            prob += Z >= (o[6] + 1) * x[(d, i)]
    links = {(min(u, v), max(u, v)) for u, v in G.edges()}
    for e in links:                                          # non-overlap (+guard)
        for sslot in range(NUM_SLOTS):
            terms = [x[(d, i)] for d in range(len(demands))
                     for i, o in enumerate(cand[d]) if e in o[1] and sslot in o[5]]
            if terms: prob += pulp.lpSum(terms) <= 1
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    assign, mx = {}, -1
    for d in range(len(demands)):
        for i, o in enumerate(cand[d]):
            if pulp.value(x[(d, i)]) > 0.5:
                assign[d] = (o[0], o[2], o[3], o[4]); mx = max(mx, o[6])
    return assign, mx

def _brute(cand, demands):                                   # verified fallback
    best = None
    for combo in product(*[range(len(c)) for c in cand]):
        used, ok = {}, True
        for d, i in enumerate(combo):
            for e in cand[d][i][1]:
                seen = used.setdefault(e, set())
                if seen & cand[d][i][5]: ok = False; break
                seen |= cand[d][i][5]
            if not ok: break
        if not ok: continue
        cost = max(cand[d][i][6] for d, i in enumerate(combo))
        if best is None or cost < best[0]: best = (cost, combo)
    if best is None: return {}, -1
    assign = {d: (cand[d][i][0], cand[d][i][2], cand[d][i][3], cand[d][i][4])
              for d, i in enumerate(best[1])}
    return assign, best[0]
```

The chain is: variable-width demands break single-wavelength colouring, so each demand becomes a contiguous slot *block* whose width comes from a distance-adaptive modulation table; contiguity is the new, fragmentation-causing constraint, and seeing assignment as fixed-machine scheduling shows the problem is NP-hard even with routes fixed; the exact route is an integer program, but explicit contiguity constraints blow it up, so I pre-enumerate contiguous *channels* and let each demand pick one — folding contiguity and continuity into the variable and leaving only a clean non-overlap-with-guard family and a min-max-index objective; for scale I decompose into route-then-First-Fit-Decreasing with a Most-Used global variant, knowing the decomposition can miss the detour-for-reuse trade that the joint program catches.
