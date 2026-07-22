# TIER: strong
# Insight: don't summarize each station into one aggregate number (net
# demand) and then size it -- follow the network's REAL, JOINT
# stockout-probability gradient directly. Starting from an empty network,
# repeatedly ask the actual replay simulator "if I had one more bike right
# now, which station would it help the most?" (i.e. compute the true
# marginal loss reduction of the NEXT bike at every station, given every
# bike already placed) and hand that bike to the winner. This is exactly
# the coupled-flow gradient the family is named for: because the simulator
# is queried with the CURRENT joint allocation each time, the answer already
# reflects cross-station coupling (a station with balanced total traffic but
# phase-separated in/out flow shows a real, large marginal gain even though
# its net demand is ~0; a station that would just overflow shows a gain of
# ~0 and stops absorbing bikes) -- something a one-shot aggregate formula
# cannot see. This is fundamentally different from greedy: greedy computes
# one static number per station and never touches the simulator; here every
# single bike is placed by re-querying the real system. A short local
# search (single-bike swaps, also against the real simulator) polishes the
# result afterward.
import sys, json, heapq


def simulate_detail(n, capacity, trips, init):
    bikes = list(init)
    order = sorted(range(len(trips)), key=lambda i: (trips[i][0], i))
    hp = []
    seq = 0
    lost = 0
    strand = 0
    for idx in order:
        t, o, d, dt = trips[idx]
        while hp and hp[0][0] <= t:
            at, sq, dest = heapq.heappop(hp)
            if bikes[dest] < capacity[dest]:
                bikes[dest] += 1
            else:
                strand += 1
        if bikes[o] > 0:
            bikes[o] -= 1
            heapq.heappush(hp, (t + dt, seq, d)); seq += 1
        else:
            lost += 1
    while hp:
        at, sq, dest = heapq.heappop(hp)
        if bikes[dest] < capacity[dest]:
            bikes[dest] += 1
        else:
            strand += 1
    return lost, strand


def sim_loss(n, capacity, trips, init, w_pickup, w_strand):
    lost, strand = simulate_detail(n, capacity, trips, init)
    return w_pickup * lost + w_strand * strand


def marginal_gradient_construct(n, capacity, budget, trips, w_pickup, w_strand):
    """Place bikes one at a time. Each placement queries the REAL joint
    simulator for every station's current marginal loss reduction (the
    coupled-flow stockout-probability gradient) and gives the bike to the
    station where that gradient is steepest right now. Stops early, leaving
    budget unspent, once no station's next bike would help at all."""
    alloc = [0] * n
    cur = sim_loss(n, capacity, trips, alloc, w_pickup, w_strand)
    for _ in range(budget):
        best_s, best_gain, best_loss = None, 1e-9, None
        for s in range(n):
            if alloc[s] >= capacity[s]:
                continue
            alloc[s] += 1
            new_loss = sim_loss(n, capacity, trips, alloc, w_pickup, w_strand)
            alloc[s] -= 1
            gain = cur - new_loss
            if gain > best_gain:
                best_s, best_gain, best_loss = s, gain, new_loss
        if best_s is None:
            break
        alloc[best_s] += 1
        cur = best_loss
    return alloc, cur


def hill_climb(n, capacity, trips, alloc, w_pickup, w_strand, iters):
    alloc = list(alloc)
    cur = sim_loss(n, capacity, trips, alloc, w_pickup, w_strand)
    for _ in range(iters):
        best_gain = 1e-9
        best_move = None
        for a in range(n):
            if alloc[a] <= 0:
                continue
            alloc[a] -= 1
            loss_new = sim_loss(n, capacity, trips, alloc, w_pickup, w_strand)
            alloc[a] += 1
            gain = cur - loss_new
            if gain > best_gain:
                best_gain = gain
                best_move = (a, None, loss_new)
            for b in range(n):
                if b == a or alloc[b] >= capacity[b]:
                    continue
                alloc[a] -= 1; alloc[b] += 1
                loss_new = sim_loss(n, capacity, trips, alloc, w_pickup, w_strand)
                alloc[a] += 1; alloc[b] -= 1
                gain = cur - loss_new
                if gain > best_gain:
                    best_gain = gain
                    best_move = (a, b, loss_new)
        if best_move is None:
            break
        a, b, loss_new = best_move
        alloc[a] -= 1
        if b is not None:
            alloc[b] += 1
        cur = loss_new
    return alloc, cur


def solve(inst):
    n = inst["n"]; cap = inst["capacity"]; budget = inst["budget"]
    trips = [tuple(t) for t in inst["trips"]]
    wp = inst["w_pickup"]; ws = inst["w_strand"]

    alloc, _ = marginal_gradient_construct(n, cap, budget, trips, wp, ws)
    alloc, _ = hill_climb(n, cap, trips, alloc, wp, ws, iters=40)
    return alloc


inst = json.load(sys.stdin)
result = solve(inst)
print(json.dumps({"init": result}))
