# TIER: strong
# Insight: the score rewards WHEN thickness is accumulated, not merely reaching
# the target. So search over fold ORDERS (beam search on the fold state), keeping
# schedules that fold thin margins first and route folds AROUND reinforced creases,
# deferring the heavy centre so it is carried as few times as possible.
import sys


def apply_fold(slots, k, h, reinforced, W):
    w = len(slots)
    pos = {}
    for si, sl in enumerate(slots):
        for c in sl:
            pos[c] = si
    bent = 0
    for c in reinforced:
        if (pos[c] < k) != (pos[c + 1] < k):
            bent += 1
    if k <= w - k:
        left = slots[:k]
        S = sum(h[c] for sl in left for c in sl)
        new = [list(sl) for sl in slots[k:]]
        for idx in range(k):
            new[k - 1 - idx].extend(left[idx])
    else:
        right = slots[k:]
        S = sum(h[c] for sl in right for c in sl)
        new = [list(sl) for sl in slots[:k]]
        for j in range(k, w):
            new[2 * k - 1 - j].extend(slots[j])
    key = tuple(tuple(sorted(sl)) for sl in new)
    return key, [tuple(sl) for sl in new], S + W * bent


def beam_search(N, T, W, h, reinforced, beam_width=400, depth_cap=60):
    start = [(c,) for c in range(N)]
    frontier = [(0, start, [])]  # (cost_so_far, slots, folds)
    best_cost = None
    best_folds = None
    depth = 0
    while frontier and depth < depth_cap:
        depth += 1
        cand = []
        seen = {}
        for cost, slots, folds in frontier:
            w = len(slots)
            for k in range(1, w):
                key, newslots, dc = apply_fold(slots, k, h, reinforced, W)
                ncost = cost + dc
                nfolds = folds + [k]
                if len(newslots) <= T:
                    if best_cost is None or ncost < best_cost:
                        best_cost = ncost
                        best_folds = nfolds
                    continue
                statekey = (len(newslots), key)
                if statekey in seen and seen[statekey] <= ncost:
                    continue
                seen[statekey] = ncost
                cand.append((ncost, newslots, nfolds))
        cand.sort(key=lambda x: x[0])
        frontier = cand[:beam_width]
    return best_cost, best_folds


def cost_of(slots0, folds, N, T, W, h, reinforced):
    slots = [list(sl) for sl in slots0]
    cost = 0
    for k in folds:
        w = len(slots)
        if not (1 <= k <= w - 1):
            return None
        _, newslots, dc = apply_fold(slots, k, h, reinforced, W)
        slots = [list(sl) for sl in newslots]
        cost += dc
    if len(slots) > T:
        return None
    return cost


def greedy_schedule(N, T):
    folds = []
    w = N
    while w > T:
        k = w // 2
        if k < 1:
            k = 1
        folds.append(k)
        w = w - min(k, w - k)
    return folds


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    N = int(next(it)); T = int(next(it)); R = int(next(it)); W = int(next(it))
    h = [int(next(it)) for _ in range(N)]
    reinforced = [int(next(it)) for _ in range(R)]

    start = [(c,) for c in range(N)]

    candidates = []
    bc, bf = beam_search(N, T, W, h, reinforced)
    if bf is not None:
        candidates.append((bc, bf))
    g = greedy_schedule(N, T)
    gc = cost_of(start, g, N, T, W, h, reinforced)
    if gc is not None:
        candidates.append((gc, g))
    acc = [1] * (N - T)
    ac = cost_of(start, acc, N, T, W, h, reinforced)
    if ac is not None:
        candidates.append((ac, acc))

    candidates.sort(key=lambda x: x[0])
    folds = candidates[0][1]
    print(len(folds))
    if folds:
        print(" ".join(str(x) for x in folds))


if __name__ == "__main__":
    main()
