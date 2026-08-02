# TIER: greedy
import sys, math


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    N = int(next(it))
    f0 = int(next(it)); a0 = int(next(it)); r0 = int(next(it))
    REV = int(next(it)); BUDGET = int(next(it)); MIN_COMPS = int(next(it))
    comps = []
    for _ in range(N):
        margin = int(next(it)); f = int(next(it)); a = int(next(it))
        r = int(next(it)); doc_cost = int(next(it))
        comps.append((margin, f, a, r, doc_cost))

    # "obvious" recipe: a loose overall-similarity screen (plain unweighted L1 distance),
    # then chase the highest reported margins, spend nothing on documentation depth,
    # declare near the top of the hand-picked set's own range.
    pool = []
    for idx in range(N):
        margin, f, a, r, doc_cost = comps[idx]
        dist = abs(f - f0) + abs(a - a0) + abs(r - r0)
        if dist < 12:
            pool.append(idx)
    if len(pool) < MIN_COMPS:
        pool = list(range(N))

    pool.sort(key=lambda idx: -comps[idx][0])
    cap = max(MIN_COMPS, min(len(pool), MIN_COMPS + 6))
    chosen = []
    spent = 0
    for idx in pool:
        if len(chosen) >= cap:
            break
        cost = comps[idx][4]  # depth 0 -> cost = doc_cost * 1
        if spent + cost <= BUDGET:
            chosen.append(idx + 1)
            spent += cost
    if len(chosen) < MIN_COMPS:
        chosen = [idx + 1 for idx in pool[:MIN_COMPS]]

    margins = sorted(comps[i1 - 1][0] for i1 in chosen)
    pos = max(0, int(math.ceil(0.75 * len(margins))) - 1)
    M = margins[pos]

    print(len(chosen))
    for i1 in chosen:
        print(i1, 0)   # no documentation spent
    print(M)


main()
