# TIER: strong
# Insight: settled material is permanent, later pours run downhill over it, and
# overshoot is charged at EVERY remaining stage.  So don't match the target now
# -- simulate the physics and price each candidate pour by how much overshoot it
# commits for the WHOLE remaining horizon.  Under-pour early, fill valleys where
# runoff stays below target, and top up only in the final stages.
import sys, random


def settle(h, S):
    N = len(h)
    while True:
        stable = True
        for i in range(N - 1):
            d = h[i] - h[i + 1]
            if d > S:
                m = (d - S + 1) // 2
                h[i] -= m; h[i + 1] += m; stable = False
            elif -d > S:
                m = (-d - S + 1) // 2
                h[i + 1] -= m; h[i] += m; stable = False
        if stable:
            return


def over(h, t):
    return sum(hi - ti for hi, ti in zip(h, t) if hi > ti)


def short(h, t):
    return sum(ti - hi for hi, ti in zip(h, t) if ti > hi)


def eval_sched(sched, N, S, L, t, K):
    h = [0] * N
    integ = 0
    for (c, g) in sched:
        h[c] += g
        settle(h, S)
        integ += over(h, t)
    return K * short(h, t) + L * integ      # matches the checker's cost exactly


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    N = int(next(it)); K = int(next(it)); S = int(next(it))
    L = int(next(it)); G = int(next(it))
    t = [int(next(it)) for _ in range(N)]

    gset = sorted(set([0, 1, G // 4, G // 3, G // 2, (2 * G) // 3, (3 * G) // 4, G]))
    gset = [g for g in gset if 0 <= g <= G]

    # ---- candidate 1: horizon-priced lookahead (price overshoot over the whole
    #      remaining horizon; under-pour early, fill valleys, defer top-ups) ----
    h = [0] * N
    look = []
    for s in range(K):
        w = K - s                      # this pour's overshoot persists for w stages
        best = None
        best_key = None
        for c in range(N):
            for g in gset:
                hh = h[:]
                hh[c] += g
                settle(hh, S)
                key = K * short(hh, t) + L * w * over(hh, t)
                if best_key is None or key < best_key:
                    best_key = key
                    best = (c, g, hh)
        c, g, hh = best
        look.append((c, g))
        h = hh

    # ---- candidate 2: the myopic per-cell recipe (so strong is never worse) --
    h = [0] * N
    grd = []
    for _ in range(K):
        c = max(range(N), key=lambda i: t[i] - h[i])
        deficit = t[c] - h[c]
        g = min(G, deficit) if deficit > 0 else 0
        grd.append((c, g))
        h[c] += g
        settle(h, S)

    cur = look if eval_sched(look, N, S, L, t, K) <= eval_sched(grd, N, S, L, t, K) else grd
    cur_cost = eval_sched(cur, N, S, L, t, K)

    # ---- local-search polish on the EXACT composed objective ----
    iters = max(120, min(400, 4000 // K))
    rng = random.Random(9091 + N * 7 + K)
    for _ in range(iters):
        j = rng.randrange(K)
        oc, og = cur[j]
        nc = rng.randrange(N)
        ng = rng.choice(gset)
        if (nc, ng) == (oc, og):
            continue
        cur[j] = (nc, ng)
        nc_cost = eval_sched(cur, N, S, L, t, K)
        if nc_cost <= cur_cost:
            cur_cost = nc_cost
        else:
            cur[j] = (oc, og)

    print("\n".join("%d %d" % (c, g) for (c, g) in cur))


main()
