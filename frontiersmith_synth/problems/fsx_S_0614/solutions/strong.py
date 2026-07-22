# TIER: strong
# INSIGHT (not "greedy + more passes"): value is NON-SEPARABLE, so pricing/rounding at the
# single-item granularity is blind to every bonus.  Three composed moves:
#   (1) SYNERGY-AWARE ROUNDING -- build candidate BLOCKS: each synergy pair (and small
#       chain-cluster) is one MERGED super-item (value v_i+v_j+s, weight w_i+w_j), competing
#       against the singleton items.  A pair enters the solution only when its JOINT ratio
#       beats the alternatives; the mediocre-ratio-but-high-bonus pairs the greedy recipe
#       ignores now surface as high-ratio super-items.
#   (2) LAGRANGIAN DUAL PRICE -- capacity is one scarce shared resource; put a price mu on
#       weight and admit a block iff value - mu*weight > 0.  Bisect mu until the admitted
#       (non-overlapping) weight fits the total capacity: the dual decides WHICH blocks --
#       and thus which pairs -- are worth the room.
#   (3) EJECTION-CHAIN REPAIR -- place admitted blocks into bins (cheapest-capacity-price =
#       emptiest bin first).  A high-value block that no bin can hold is admitted by EJECTING
#       the lowest-value occupants, which then relocate to another bin -- a short chain that
#       repairs the packing without a full re-solve.  A final co-location pass pulls apart
#       pairs the packing split, when moving one side improves the objective.
import sys, json

inst = json.load(sys.stdin)
N, M = inst["N"], inst["M"]
C = list(inst["C"])
w = [float(x) for x in inst["w"]]
v = [float(x) for x in inst["v"]]
syn = inst["syn"]
cap_max = max(C) if C else 0.0

# synergy adjacency
adj = [[] for _ in range(N)]
pair_s = {}
for (i, j, s) in syn:
    adj[i].append((j, float(s)))
    adj[j].append((i, float(s)))
    pair_s[(min(i, j), max(i, j))] = float(s)


def sval(a, b):
    return pair_s.get((min(a, b), max(a, b)), 0.0)


# ---- (1) build candidate blocks: singletons + pair super-items + size<=3 clusters ----
blocks = []  # (items tuple, value, weight)
for i in range(N):
    blocks.append(((i,), v[i], w[i]))
for (i, j, s) in syn:
    wt = w[i] + w[j]
    if wt <= cap_max + 1e-9:
        blocks.append(((i, j), v[i] + v[j] + float(s), wt))
# small clusters (chains/triangles) of size 3 that still fit one bin
seen_tri = set()
for i in range(N):
    for (j, sij) in adj[i]:
        for (k, sjk) in adj[j]:
            if k == i:
                continue
            tri = tuple(sorted((i, j, k)))
            if tri in seen_tri:
                continue
            seen_tri.add(tri)
            wt = w[i] + w[j] + w[k]
            if wt <= cap_max + 1e-9:
                val = v[i] + v[j] + v[k] + sval(i, j) + sval(j, k) + sval(i, k)
                blocks.append((tri, val, wt))

totC = float(sum(C))


# ---- (2) Lagrangian selection: given price mu, pick non-overlapping blocks with
#          positive reduced value, by descending ratio; return chosen + total weight. ----
def select(mu):
    cand = []
    for (items, val, wt) in blocks:
        red = val - mu * wt
        if red > 1e-9 and wt > 0:
            cand.append((val / wt, val, wt, items))
    cand.sort(key=lambda t: (-t[0], -t[1]))
    used = [False] * N
    chosen = []
    tot = 0.0
    for (ratio, val, wt, items) in cand:
        if any(used[it] for it in items):
            continue
        for it in items:
            used[it] = True
        chosen.append((items, val, wt))
        tot += wt
    return chosen, tot


# bisect mu so admitted weight is just within total capacity (dual pricing of capacity)
lo, hi = 0.0, 0.0
for (items, val, wt) in blocks:
    if wt > 0:
        hi = max(hi, val / wt)
hi *= 1.05
chosen, tot = select(lo)
if tot > totC:
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        chosen, tot = select(mid)
        if tot > totC:
            lo = mid
        else:
            hi = mid
    chosen, tot = select(hi)

# order admitted blocks by value density then value (rounding order)
chosen.sort(key=lambda b: (-(b[1] / b[2]) if b[2] > 0 else 0.0, -b[1]))

# ---- (3) place blocks; eject low-value occupants when a block cannot fit ----
assign = [-1] * N
rem = list(map(float, C))
# track which items sit in each bin for ejection
bin_items = [[] for _ in range(M)]


def place_block(items, wt):
    # worst-fit: emptiest bin that fits = cheapest capacity price
    best, best_rem = -1, -1.0
    for b in range(M):
        if rem[b] >= wt - 1e-9 and rem[b] > best_rem:
            best, best_rem = b, rem[b]
    if best >= 0:
        for it in items:
            assign[it] = best
            bin_items[best].append(it)
        rem[best] -= wt
        return True
    return False


for (items, val, wt) in chosen:
    if place_block(items, wt):
        continue
    # ejection chain: find the bin where freeing room for this block is cheapest.
    # eject lowest-value singletons (value < this block) and try to relocate them.
    best_bin, best_cost, best_evict = -1, None, None
    for b in range(M):
        need = wt - rem[b]
        if need <= 0:
            continue
        occ = sorted(bin_items[b], key=lambda it: v[it])  # cheapest first
        freed, evict, cost = 0.0, [], 0.0
        for it in occ:
            if freed >= need:
                break
            evict.append(it)
            freed += w[it]
            cost += v[it]
        if freed >= need - 1e-9 and cost < val:
            if best_cost is None or cost < best_cost:
                best_bin, best_cost, best_evict = b, cost, evict
    if best_bin >= 0:
        b = best_bin
        for it in best_evict:
            assign[it] = -1
            bin_items[b].remove(it)
            rem[b] += w[it]
        for it in items:
            assign[it] = b
            bin_items[b].append(it)
        rem[b] -= wt
        # relocate ejected items into any bin with room (one hop of the chain)
        for it in sorted(best_evict, key=lambda x: -v[x]):
            for bb in range(M):
                if rem[bb] >= w[it] - 1e-9:
                    assign[it] = bb
                    bin_items[bb].append(it)
                    rem[bb] -= w[it]
                    break

# ---- final co-location pass: for a split synergy pair, move the smaller side into the
#      partner's bin (directly or by ejecting a cheaper occupant) if obj improves. ----
def obj_now():
    o = 0.0
    for i in range(N):
        if assign[i] != -1:
            o += v[i]
    for (i, j, s) in syn:
        if assign[i] != -1 and assign[i] == assign[j]:
            o += float(s)
    return o


for _pass in range(2):
    for (i, j, s) in syn:
        bi, bj = assign[i], assign[j]
        if bi == -1 or bj == -1 or bi == bj:
            continue
        # try moving i into bj (or j into bi); accept only if the whole objective improves
        # (moving an item can shed synergies it had in its old bin).
        for (mv, tgt) in ((i, bj), (j, bi)):
            if rem[tgt] >= w[mv] - 1e-9:
                src = assign[mv]
                before = obj_now()
                assign[mv] = tgt
                rem[src] += w[mv]; rem[tgt] -= w[mv]
                bin_items[src].remove(mv); bin_items[tgt].append(mv)
                if obj_now() <= before + 1e-9:  # revert
                    assign[mv] = src
                    rem[src] -= w[mv]; rem[tgt] += w[mv]
                    bin_items[tgt].remove(mv); bin_items[src].append(mv)
                else:
                    break

# ---- fill leftover capacity with the best remaining singletons (never waste a bin) ----
left = [i for i in range(N) if assign[i] == -1]
left.sort(key=lambda i: (v[i] / w[i] if w[i] > 0 else 0.0), reverse=True)
for i in left:
    best, best_rem = -1, -1.0
    for b in range(M):
        if rem[b] >= w[i] - 1e-9 and rem[b] > best_rem:
            best, best_rem = b, rem[b]
    if best >= 0:
        assign[i] = best
        rem[best] -= w[i]
        bin_items[best].append(i)

print(json.dumps({"assign": assign}))
