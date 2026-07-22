# TIER: strong
# The insight: changeover cost is NOT a fixed matrix -- it is the outcome of a
# continuous dilution process, and because the flush formula depends on the TARGET
# colour's own purity threshold (not the pair as a symmetric "distance"), the direct
# cost matrix does not obey its own triangle inequality.  Computing the ALL-PAIRS
# shortest path over that matrix (Floyd-Warshall) exposes an effective changeover
# graph where routing i -> k -> j can beat the direct edge i -> j whenever k's own
# spec is loose.  Realising a shortcut costs a real extra campaign (perturbing every
# colour's inventory through the shared cycle time), so we only ever accept an
# insertion when it demonstrably lowers the SAME total-cost objective the evaluator
# uses (self-consistent lot sizing recomputed after every candidate move) -- this is
# a local search over the true objective, not a static formula.  We seed it from
# several nearest-neighbour tours on the shortest-path (not direct) matrix, refine
# with 2-opt and single-colour relocation (also scored on the true objective), and
# only then try expanding tour edges into their shortest paths, iterating until
# nothing helps.
import sys, json, math

inst = json.load(sys.stdin)
colors = inst["colors"]
k = inst["k"]
lam = inst["lambda"]
flush_cost = inst["flush_cost"]
waste_price = inst["waste_price"]
cycles = inst["cycles"]
max_lot = inst["max_lot"]
max_campaigns = inst["max_campaigns"]

tints = [c["tint"] for c in colors]
taus = [c["tau"] for c in colors]
demand = [c["demand"] for c in colors]
hold = [c["hold"] for c in colors]
back = [c["back"] for c in colors]
minlot = [c["min_lot"] for c in colors]


def waste(i, j):
    if i == j:
        return 0
    diff = abs(tints[i] - tints[j])
    if diff <= taus[j]:
        return 0
    ratio = diff / float(taus[j])
    steps = math.log(ratio) / math.log(1.0 / lam)
    return flush_cost * max(0, int(math.ceil(steps - 1e-9)))


W = [[waste(i, j) for j in range(k)] for i in range(k)]

# ---- all-pairs shortest path over the direct dilution-cost graph -----------
INF = float("inf")
dist = [row[:] for row in W]
nxt = [[j for j in range(k)] for _ in range(k)]
for m in range(k):
    dm = dist[m]
    for i in range(k):
        dim = dist[i][m]
        if dim == INF:
            continue
        di = dist[i]
        for j in range(k):
            cand = dim + dm[j]
            if cand < di[j] - 1e-12:
                di[j] = cand
                nxt[i][j] = nxt[i][m]


def sp_path(i, j):
    p = [i]
    guard = 0
    while i != j and guard < 4 * k:
        i = nxt[i][j]
        p.append(i)
        guard += 1
    return p


# ---- timeline / self-consistent lot sizing / true objective ----------------
def timeline(order, lots):
    t = 0
    prev = order[-1]
    events = {}
    totw = 0
    for idx, c in enumerate(order):
        w = waste(prev, c)
        totw += w
        t += w
        events.setdefault(c, []).append((t, lots[idx]))
        t += lots[idx]
        prev = c
    return t, events, totw


def self_consistent(order, iters=8):
    Test = 300.0
    lots = None
    for _ in range(iters):
        lots = []
        for c in order:
            L = max(minlot[c], int(round(demand[c] * Test)))
            L = min(L, max_lot)
            lots.append(L)
        Test, _, _ = timeline(order, lots)
    return lots


def integrate_linear(level0, rate, length, h, p):
    if length <= 0:
        return 0.0
    if rate <= 1e-15:
        lvl = level0
        return length * (h * max(0.0, lvl) + p * max(0.0, -lvl))
    tstar = level0 / rate
    if level0 >= 0:
        t1 = min(max(tstar, 0.0), length)
        pos = h * (level0 * t1 - 0.5 * rate * t1 * t1)
        neg = 0.0
        if t1 < length:
            t2 = length
            neg = p * (0.5 * rate * (t2 * t2 - t1 * t1) - level0 * (t2 - t1))
        return pos + neg
    else:
        t2 = length
        return p * (0.5 * rate * (t2 * t2) - level0 * t2)


def cost_of_lots(order, lots):
    T_cyc, events, wpc = timeline(order, lots)
    if T_cyc <= 0:
        return float("inf")
    horizon_end = cycles * T_cyc
    cost = waste_price * wpc * cycles
    for cid in range(k):
        local = events.get(cid, [])
        d = demand[cid]; h = hold[cid]; p = back[cid]
        level = 0.0
        cur_t = 0.0
        cc = 0.0
        for r in range(cycles):
            base = r * T_cyc
            for (loc_t, lot) in local:
                et = base + loc_t
                seg = et - cur_t
                if seg > 0:
                    cc += integrate_linear(level, d, seg, h, p)
                    level -= d * seg
                level += lot
                cur_t = et
        seg = horizon_end - cur_t
        if seg > 0:
            cc += integrate_linear(level, d, seg, h, p)
        cost += cc
    return cost


def lot_search(order, lots):
    """Coordinate-wise refinement of each campaign's lot size against the TRUE
    (real, non-reset) multi-cycle objective: demand-matched sizing is a good
    starting guess, but is not always optimal -- shrinking a lot trades some
    backlog for a shorter cycle (helping every colour's review period), and
    growing one trades holding cost for fewer changeovers elsewhere. Try both
    directions and keep whichever the real objective prefers. (Expensive -- only
    run this ONCE, on the final tour, not inside the order search below.)"""
    lots = list(lots)
    cur = cost_of_lots(order, lots)
    for _ in range(3):
        improved = False
        for idx, c in enumerate(order):
            base = lots[idx]
            candidates = {minlot[c], max_lot}
            for frac in (0.4, 0.6, 0.8, 1.2, 1.5, 2.0):
                candidates.add(max(minlot[c], min(max_lot, int(round(base * frac)))))
            for L in candidates:
                if L == lots[idx]:
                    continue
                trial = list(lots)
                trial[idx] = L
                c2 = cost_of_lots(order, trial)
                if c2 < cur - 1.0:
                    lots = trial
                    cur = c2
                    improved = True
        if not improved:
            break
    return lots, cur


def total_cost(order):
    """Cheap objective (self-consistent demand-matched lots only) used while
    searching over TOUR ORDER -- fast enough to call thousands of times."""
    if len(order) < 1 or len(order) > max_campaigns:
        return float("inf"), None
    lots = self_consistent(order)
    c = cost_of_lots(order, lots)
    return c, lots


def cost_only(order):
    c, _ = total_cost(order)
    return c


def nn_tour(cost_matrix, start):
    unvisited = set(range(k))
    cur = start
    unvisited.discard(cur)
    tour = [cur]
    while unvisited:
        n = min(unvisited, key=lambda j: cost_matrix[cur][j])
        tour.append(n)
        unvisited.discard(n)
        cur = n
    return tour


def two_opt(tour):
    cur = cost_only(tour)
    improved = True
    while improved:
        improved = False
        n = len(tour)
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]
                c = cost_only(cand)
                if c < cur - 1.0:
                    tour = cand
                    cur = c
                    improved = True
    return tour, cur


def or_opt(tour, cur):
    improved = True
    while improved:
        improved = False
        n = len(tour)
        for i in range(n):
            city = tour[i]
            rest = tour[:i] + tour[i + 1:]
            for pos in range(len(rest) + 1):
                cand = rest[:pos] + [city] + rest[pos:]
                if cand == tour:
                    continue
                c = cost_only(cand)
                if c < cur - 1.0:
                    tour = cand
                    cur = c
                    improved = True
                    break
            if improved:
                break
    return tour, cur


# multi-start NN seeded from the SHORTEST-PATH matrix + local search on the TRUE
# objective (this is where the routing insight actually gets used: nn_tour explores
# orderings informed by effective, catalyst-aware costs, not raw direct edges)
best_tour, best_cost = None, float("inf")
for s in range(min(k, 6)):
    t = nn_tour(dist, s)
    t, c = two_opt(t)
    if c < best_cost:
        best_tour, best_cost = t, c
tour, cur_cost = best_tour, best_cost

# selective catalyst insertion: only expand a tour edge into its shortest PATH
# (inserting the real intermediate campaigns) if that strictly lowers the true cost
for _ in range(3):
    improved = False
    idx = 0
    while idx < len(tour):
        i = tour[idx]
        j = tour[(idx + 1) % len(tour)]
        p = sp_path(i, j)
        if len(p) > 2 and len(tour) + len(p) - 2 <= max_campaigns:
            cand = tour[:idx + 1] + p[1:-1] + tour[idx + 1:]
            c = cost_only(cand)
            if c < cur_cost - 1.0:
                tour = cand
                cur_cost = c
                improved = True
        idx += 1
    if not improved:
        break
    tour, cur_cost = two_opt(tour)

tour, cur_cost = or_opt(tour, cur_cost)
tour, cur_cost = two_opt(tour)

# final refinement: once the ORDER is settled, spend the extra search budget on
# genuinely optimizing lot sizes against the true objective (not just matching
# demand) -- this is what keeps strong from being beaten by a "ship every colour
# the bare minimum" degenerate policy on instances where a shorter cycle wins.
_, lots0 = total_cost(tour)
lots, _ = lot_search(tour, lots0)

wheel = [{"color": tour[idx], "lot": lots[idx]} for idx in range(len(tour))]
print(json.dumps({"wheel": wheel}))
