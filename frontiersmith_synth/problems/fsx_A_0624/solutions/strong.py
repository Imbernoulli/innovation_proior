# TIER: strong
# INSIGHT: capacity is a lever on deferred-acceptance's rejection-chain topology,
# so the objective is violently non-local in the quotas -- demand counts do NOT
# predict a seat's marginal value (a workshop nobody lists FIRST can still be the
# only lifeline that rescues an entire cascade of rejected apprentices, and a
# workshop with huge first-choice demand can be mostly locked to a small elite,
# so extra seats there barely move the total). Instead of a static demand proxy,
# treat the DA run itself as the object of optimization: try a few structurally
# different seed allocations (raw demand share, uniform, size-concentrated),
# then do verified coordinate-ascent seat transfers guided by tracing WHICH
# workshop each poorly-off apprentice was rejected from, always re-running DA to
# confirm a transfer actually helps. Deterministic: fixed iteration budget.
import sys, heapq


def run_da(N, M, prefs, prio_rank, caps):
    next_choice = [0] * N
    held = [[] for _ in range(M)]
    match = [-1] * N
    stack = list(range(N))
    while stack:
        s = stack.pop()
        pl = prefs[s]; nc = next_choice[s]; Ls = len(pl)
        while nc < Ls:
            wk = pl[nc]; nc += 1
            if caps[wk] <= 0:
                continue
            r = prio_rank[wk][s]; h = held[wk]
            if len(h) < caps[wk]:
                heapq.heappush(h, (-r, s)); match[s] = wk
                break
            worst_negr, worst_s = h[0]
            if r < -worst_negr:
                heapq.heapreplace(h, (-r, s))
                match[s] = wk
                match[worst_s] = -1
                pw = prefs[worst_s]
                np_ = len(pw)
                for idx in range(np_):
                    if pw[idx] == wk:
                        next_choice[worst_s] = idx + 1
                        break
                else:
                    next_choice[worst_s] = np_
                stack.append(worst_s)
                break
        next_choice[s] = nc
    return match


def evaluate(N, M, prefs, prio_rank, caps, W):
    match = run_da(N, M, prefs, prio_rank, caps)
    U = 0
    for s in range(N):
        wk = match[s]
        if wk < 0:
            continue
        pl = prefs[s]
        for idx in range(len(pl)):
            if pl[idx] == wk:
                U += W[idx]
                break
    return U, match


def local_search(N, M, prefs, prio_rank, cap_max, W, start, budget, topk=10):
    caps = start[:]
    curU, _ = evaluate(N, M, prefs, prio_rank, caps, W)
    best = caps[:]; bestU = curU
    for _ in range(budget):
        U, match = evaluate(N, M, prefs, prio_rank, caps, W)
        held = [0] * M
        for s in range(N):
            if match[s] >= 0:
                held[match[s]] += 1
        donors = sorted((w for w in range(M) if caps[w] > held[w]),
                         key=lambda w: -(caps[w] - held[w]))
        if not donors:
            donors = sorted((w for w in range(M) if caps[w] > 0), key=lambda w: caps[w])
        loss = [0] * M
        for s in range(N):
            wk = match[s]
            pl = prefs[s]
            if wk < 0:
                loss[pl[0]] += W[0]
                continue
            best_idx = 0
            for idx, w in enumerate(pl):
                if w == wk:
                    best_idx = idx
                    break
            if best_idx > 0:
                loss[pl[0]] += (W[0] - W[best_idx])
        receivers = sorted(range(M), key=lambda w: -loss[w])
        improved = False
        for r in receivers[:topk]:
            if caps[r] >= cap_max[r] or loss[r] == 0:
                continue
            for d in donors[:topk]:
                if d == r or caps[d] <= 0:
                    continue
                caps[d] -= 1; caps[r] += 1
                nU, _ = evaluate(N, M, prefs, prio_rank, caps, W)
                if nU > curU:
                    curU = nU; improved = True
                    break
                caps[d] += 1; caps[r] -= 1
            if improved:
                break
        if not improved:
            break
        if curU > bestU:
            bestU = curU; best = caps[:]
    return best, bestU


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it)); T = int(next(it))
    cap_max = [int(next(it)) for _ in range(M)]
    prefs = []
    for _ in range(N):
        Li = int(next(it))
        prefs.append([int(next(it)) for _ in range(Li)])
    order = [[int(next(it)) for _ in range(N)] for _ in range(M)]

    W = [100, 45, 8, 3, 1]

    prio_rank = [[0] * N for _ in range(M)]
    for wk in range(M):
        ow = order[wk]; pr = prio_rank[wk]
        for pos in range(N):
            pr[ow[pos]] = pos

    # ---- candidate seed allocations ----
    uni = [T // M] * M
    for j in range(T % M):
        uni[j] += 1

    demand = [0] * M
    for pl in prefs:
        demand[pl[0]] += 1
    tot = sum(demand)
    dem = [0] * M
    if tot > 0:
        raw = [T * demand[w] / tot for w in range(M)]
        dem = [min(cap_max[w], int(raw[w])) for w in range(M)]
    rem = T - sum(dem)
    guard = 0
    while rem > 0 and guard < 10 * M + 20:
        guard += 1
        cands = [w for w in range(M) if dem[w] < cap_max[w]]
        if not cands:
            break
        cands.sort(key=lambda w: (-demand[w], w))
        given = 0
        for w in cands:
            if rem <= 0:
                break
            dem[w] += 1; rem -= 1; given += 1
        if given == 0:
            break

    # concentrated-by-capacity seed: fill the highest-room workshops first
    order_w = sorted(range(M), key=lambda w: -cap_max[w])
    conc = [0] * M
    remc = T
    for w in order_w:
        take = min(cap_max[w], remc)
        conc[w] = take; remc -= take
        if remc <= 0:
            break

    budget = max(40, min(200, 4000 // max(1, M)))

    best_all = None; bestU_all = -1
    for seed in (dem, uni, conc):
        b, u = local_search(N, M, prefs, prio_rank, cap_max, W, seed, budget)
        if u > bestU_all:
            bestU_all = u; best_all = b

    sys.stdout.write("\n".join(map(str, best_all)) + "\n")


if __name__ == "__main__":
    main()
