# TIER: strong
# Probe + reserve-headroom + timed repack-consolidation.
#
# Insight: opening a new truck is free until it holds cargo -- the real
# opportunity cost is TOPPING OFF an existing truck with a second medium
# parcel, which permanently forecloses that truck's leftover capacity for a
# possible oversized parcel later. So during the "cautious" phase (before a
# shift looks likely), the FIRST few times normal best-fit would have to open
# a new truck anyway, some of those new trucks are deliberately held out of
# further best-fit circulation ("reserved") instead of being topped off by
# the next medium parcel -- at zero extra truck cost, since a reserved truck
# already holds exactly what best-fit would have given it regardless.
#
# The running ratio of a short recent window's mean size to the frozen probe
# mean is the live distribution estimate: once it jumps, a shift is
# confirmed, so (a) the reserved trucks rejoin the pool (their slack now
# absorbs oversized parcels directly, often with zero repacking) and (b) the
# repack budget is spent immediately, greedily pairing the lightest and
# heaviest currently-open trucks whose combined load still fits one truck
# (classic two-pointer matching) to reclaim more free trucks before the rest
# of the shift's parcels arrive. If no shift is ever confirmed, the reserved
# trucks are released back for ordinary use partway through so the caution
# doesn't cost anything on a day that stays calm; any leftover budget is
# spent on one final cleanup pass. No single piece here is just "best-fit
# with a tuned constant" -- probing, reservation, and timed repack all have
# to work together for the composite gain.
import sys, json

inst = json.load(sys.stdin)
C = inst["capacity"]
sizes = inst["sizes"]
n = inst["n"]
budget = inst["repack_budget"]

P = max(6, n // 6)              # probe window before the estimate is trusted
W = 7                            # sliding recent-window size
RATIO_TRIG = 1.6                 # recent/probe mean ratio that confirms a shift
K = max(1, round(n * 0.20))      # how many trucks may be held in reserve
RELEASE_AT = int(0.60 * n)       # give up reserving if no shift confirmed by here

loads = []            # cumulative load per opened truck
normal_pool = []      # truck ids ordinary best-fit may use
reserved_pool = []    # truck ids held back, pending a confirmed shift
assign = [0] * n
moves = []
moves_left = budget


def best_fit_among(pool, s):
    best = -1
    best_free = None
    for b in pool:
        if loads[b] + s <= C:
            free = C - loads[b] - s
            if best == -1 or free < best_free:
                best = b
                best_free = free
    return best


def try_place(i, s, burst):
    pool = normal_pool + reserved_pool if burst else normal_pool
    best = best_fit_among(pool, s)
    if best != -1:
        loads[best] += s
        assign[i] = best
        return
    new_id = len(loads)
    loads.append(0)
    loads[new_id] += s
    assign[i] = new_id
    if (not burst) and len(reserved_pool) < K:
        reserved_pool.append(new_id)
    else:
        normal_pool.append(new_id)


def consolidate(after, placed_upto, budget_frac):
    """Two-pointer matching on sorted loads: pair the lightest open truck with
    the heaviest one it still fits inside, moving whichever side has fewer
    parcels (cheaper). Spends at most budget_frac of the remaining budget."""
    global moves_left
    spend_cap = int(moves_left * budget_frac)
    if spend_cap <= 0:
        return
    active = sorted([b for b in range(len(loads)) if loads[b] > 0], key=lambda b: loads[b])
    if len(active) < 2:
        return
    lo, hi = 0, len(active) - 1
    spent = 0
    while lo < hi:
        a, b = active[lo], active[hi]
        if loads[a] == 0:
            lo += 1
            continue
        if loads[b] == 0:
            hi -= 1
            continue
        if loads[a] + loads[b] <= C:
            items_a = [k for k in range(placed_upto) if assign[k] == a]
            items_b = [k for k in range(placed_upto) if assign[k] == b]
            if len(items_a) <= len(items_b):
                src, dst, items_src = a, b, items_a
            else:
                src, dst, items_src = b, a, items_b
            cost = len(items_src)
            if cost == 0 or spent + cost > spend_cap:
                hi -= 1
                continue
            for k in items_src:
                moves.append({"after": after, "item": k, "to": dst})
                assign[k] = dst
            loads[dst] += loads[src]
            loads[src] = 0
            spent += cost
            lo += 1
            hi -= 1
        else:
            hi -= 1
    moves_left -= spent


probe_mean = sum(sizes[:P]) / P
burst_detected = False
released = False
for t in range(n):
    s = sizes[t]
    if not burst_detected and t >= P + W - 1:
        recent_mean = sum(sizes[t - W + 1:t + 1]) / W
        if probe_mean > 0 and recent_mean / probe_mean >= RATIO_TRIG:
            burst_detected = True
            consolidate(after=t - 1, placed_upto=t, budget_frac=0.75)
    if (not burst_detected) and (not released) and t >= RELEASE_AT and reserved_pool:
        normal_pool.extend(reserved_pool)
        reserved_pool = []
        released = True
    try_place(t, s, burst_detected)

consolidate(after=n - 1, placed_upto=n, budget_frac=1.0)

print(json.dumps({"placements": assign, "moves": moves}))
