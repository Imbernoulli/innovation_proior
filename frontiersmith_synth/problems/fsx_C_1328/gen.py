#!/usr/bin/env python3
"""gen.py <testId> -- prints one thermostability mutation-design instance to stdout.
Deterministic: all randomness is seeded purely from testId.
"""
import sys
import random


def main():
    test_id = int(sys.argv[1])
    rng = random.Random(20260 + test_id * 97)

    # difficulty ladder: n candidate sites, K mutation budget, C crowd cap near active site
    n_list = [8, 9, 10, 11, 12, 13, 14, 15, 16, 18]
    k_list = [3, 3, 4, 4, 4, 5, 5, 5, 5, 6]
    cap_list = [2, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    # trap cases: >=3 of the 10 plant an epistasis+crowding trap that ambushes
    # naive "rank by individual stability delta, then stack" selection.
    trap_list = [False, False, True, True, True, True, True, True, True, True]

    idx = test_id - 1
    n = n_list[idx]
    K = k_list[idx]
    C = cap_list[idx]
    R = 3            # residues at structural distance <= R are "active-site neighbourhood"
    alpha = 4.0       # crowd-penalty coefficient (quadratic in neighbourhood overcrowding)
    A0 = 100.0        # wild-type activity
    ActMin = 55.0     # minimum tolerable activity (the tradeoff floor)
    is_trap = trap_list[idx]

    dstab = [0.0] * n
    dact = [0.0] * n
    dist = [0] * n
    for i in range(n):
        dstab[i] = round(rng.uniform(1.0, 4.0), 3)
        # activity-stability tradeoff: larger individual stability gains tend to cost
        # more activity, plus independent noise
        dact[i] = round(-(0.55 * dstab[i] + rng.uniform(0.0, 2.5)), 3)
        dist[i] = rng.randint(0, 9)

    order = sorted(range(n), key=lambda i: -dstab[i])

    epi = {}

    def add_epi(i, j, es, ea):
        a, b = (i, j) if i < j else (j, i)
        epi[(a, b)] = [round(es, 3), round(ea, 3)]

    # background epistasis noise (small magnitude, scattered pairs): most pairs are NOT
    # tabulated at all (assumed additive); a sparse few carry a mild correction.
    n_bg = min(n * (n - 1) // 2, max(3, n // 3))
    tried = set()
    tries = 0
    while len(tried) < n_bg and tries < 10 * n_bg + 20:
        tries += 1
        i, j = rng.sample(range(n), 2)
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in tried:
            continue
        tried.add((a, b))
        add_epi(a, b, rng.uniform(-0.6, 0.6), rng.uniform(-1.0, 1.0))

    if is_trap and K >= 2 and n >= K + 3:
        t0, t1 = order[0], order[1]
        # push the two top individually-ranked mutations into the active-site neighbourhood
        dist[t0] = min(dist[t0], R)
        dist[t1] = min(dist[t1], R)
        # planted negative epistasis: individually the two best moves, but together they
        # clash structurally and wipe out almost all of the combined stability gain
        add_epi(t0, t1, -0.95 * (dstab[t0] + dstab[t1]), rng.uniform(-2.0, -0.6))
        if K >= 3:
            t2 = order[2]
            add_epi(t0, t2, -0.5 * (dstab[t0] + dstab[t2]), rng.uniform(-1.0, 0.0))
        # planted positive (synergistic) epistasis on a pair that individually ranks just
        # OUTSIDE the naive top-K cut, so rank-and-stack never even tries them together
        g1, g2 = order[K], order[K + 1]
        add_epi(g1, g2, 0.85 * (dstab[g1] + dstab[g2]), rng.uniform(0.0, 1.5))
        # keep the synergy pair individually affordable in activity so it is a genuinely
        # reachable, non-trivial improvement once discovered
        dact[g1] = round(-(0.30 * dstab[g1] + rng.uniform(0.0, 0.8)), 3)
        dact[g2] = round(-(0.30 * dstab[g2] + rng.uniform(0.0, 0.8)), 3)

    print(n, K, C, R)
    print("%.3f %.3f %.3f" % (A0, ActMin, alpha))
    for i in range(n):
        print("%.3f %.3f %d" % (dstab[i], dact[i], dist[i]))
    print(len(epi))
    for (i, j), (es, ea) in sorted(epi.items()):
        print(i, j, "%.3f" % es, "%.3f" % ea)


if __name__ == "__main__":
    main()
