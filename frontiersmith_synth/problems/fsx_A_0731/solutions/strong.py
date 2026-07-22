# TIER: strong
#!/usr/bin/env python3
"""Insight: the storm sweep's worst case is driven by finite-volume
fill-and-spill, not raw depth. A basin only floods a neighbor once its own
water rises past the ridge (saddle) between them -- so a cheap raise at the
right saddle can stop an entire overflow from ever reaching a valuable
cluster, protecting many cells for the price of one. Repeatedly re-simulate
the *whole* 24-storm sweep (same fill-and-spill physics as the checker) at
several step sizes to find a cell -- a ridge/saddle or a directly-flooded
cell -- that buys a genuine (positive) drop in worst-case damage, spend a
chunk there, and repeat; only ever commit moves that actually help. Falls
back to direct cell-by-cell protection (the greedy recipe) for whatever
budget the chokepoint search can't usefully place, and finally keeps
whichever of the two full allocations scores better -- so this can never
lose to the naive recipe, only beat it where the sweep's topology matters."""
import sys

NEG = -10 ** 9
EPS = 1e-9


def _cascade(lyr, pending, W):
    score_added = 0.0
    while len(lyr) > 1 and pending > 0:
        lvl0, w0, sv0 = lyr[0]
        lvl1, w1, sv1 = lyr[1]
        room = lvl1 - lvl0
        if room < 0:
            room = 0.0
        cap = w0 * room
        if pending >= cap:
            score_added += sv0 * room
            pending -= cap
            lyr[0] = [lvl1, w0 + w1, sv0 + sv1]
            del lyr[1]
        else:
            rise = pending / w0 if w0 > 0 else 0.0
            score_added += sv0 * rise
            lyr[0][0] += rise
            pending = 0.0
    if len(lyr) == 1 and pending > 0:
        lvl0, w0, sv0 = lyr[0]
        room = W - lvl0
        if room > 0:
            cap = w0 * room
            if pending >= cap:
                score_added += sv0 * room
                pending -= cap
                lyr[0] = [W, w0, sv0]
            else:
                rise = pending / w0 if w0 > 0 else 0.0
                score_added += sv0 * rise
                lyr[0][0] += rise
                pending = 0.0
    return pending, score_added


def objective(H, v, storms, N):
    best = 0.0
    for (a, b, V) in storms:
        span = b - a + 1
        base = V // span
        rem = V - base * span
        inflow = [0] * N
        for i in range(a, b + 1):
            inflow[i] = base
        for i in range(a, a + rem):
            inflow[i] += 1

        M = N + 2
        parent = list(range(M))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        pending = [0.0] * M
        is_sea = [False] * M
        layers = [None] * M
        for p in range(1, N + 1):
            layers[p] = [[float(H[p - 1]), 1, float(v[p - 1])]]
            pending[p] = float(inflow[p - 1])
        is_sea[0] = True
        is_sea[N + 1] = True
        layers[0] = []
        layers[N + 1] = []

        def ext_h(p):
            if p == 0 or p == N + 1:
                return NEG
            return H[p - 1]

        edges = []
        for p in range(0, M - 1):
            w = ext_h(p)
            w2 = ext_h(p + 1)
            if w2 > w:
                w = w2
            edges.append((w, p))
        edges.sort(key=lambda t: (t[0], t[1]))

        score = 0.0
        for (W, p) in edges:
            ri, rj = find(p), find(p + 1)
            if is_sea[ri] and is_sea[rj]:
                new_is_sea = True
                new_layers = []
                new_pending = 0.0
            else:
                combined_layers = layers[ri] + layers[rj]
                combined_layers.sort()
                combined_pending = pending[ri] + pending[rj]
                new_pending, added = _cascade(combined_layers, combined_pending, W)
                score += added
                new_layers = combined_layers
                was_sea = is_sea[ri] or is_sea[rj]
                if was_sea and len(new_layers) == 1 and new_layers[0][0] >= W - EPS:
                    new_is_sea = True
                    new_layers = []
                    new_pending = 0.0
                else:
                    new_is_sea = False

            parent[ri] = rj
            is_sea[rj] = new_is_sea
            layers[rj] = new_layers
            pending[rj] = new_pending

        if score > best:
            best = score
    return best


def historical_fd(e, v, storms, N):
    """Naive infinite-rain exposure proxy, same recipe as greedy.py."""
    fd = [0] * N
    LOCAL_CAP = 20
    for (a, b, V) in storms:
        m = b - a + 1
        avg_rain = V / max(1, min(m, LOCAL_CAP))
        pref = [0] * m
        cur = -10 ** 9
        for idx in range(a, b + 1):
            if e[idx] > cur:
                cur = e[idx]
            pref[idx - a] = cur
        suf = [0] * m
        cur = -10 ** 9
        for idx in range(b, a - 1, -1):
            if e[idx] > cur:
                cur = e[idx]
            suf[idx - a] = cur
        for idx in range(a, b + 1):
            ceiling = min(pref[idx - a], suf[idx - a])
            depth = ceiling - e[idx]
            if avg_rain < depth:
                depth = avg_rain
            if depth > fd[idx]:
                fd[idx] = int(depth)
    return fd


def greedy_fill(e, v, storms, N, h, budget):
    """Direct cell-by-cell protection using the CURRENT (already-leveed)
    terrain e[i]+h[i] as the base, spending `budget` more."""
    cur_e = [e[i] + h[i] for i in range(N)]
    fd = historical_fd(cur_e, v, storms, N)
    order = sorted(range(N), key=lambda i: fd[i] * v[i], reverse=True)
    remaining = budget
    for i in order:
        if remaining <= 0:
            break
        need = fd[i]
        if need <= 0:
            continue
        give = min(need, remaining)
        h[i] += give
        remaining -= give
    return remaining


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    Budget = int(next(it))
    K = int(next(it))
    e = [int(next(it)) for _ in range(N)]
    v = [int(next(it)) for _ in range(N)]
    storms = []
    for _ in range(K):
        a = int(next(it))
        b = int(next(it))
        V = int(next(it))
        storms.append((a, b, V))

    # --- baseline: pure greedy (guaranteed floor) ---
    h_greedy = [0] * N
    greedy_fill(e, v, storms, N, h_greedy, Budget)

    # --- smart: multi-scale chokepoint search, only commit true improvements ---
    h_smart = [0] * N
    remaining = Budget
    MAX_CAND = 24

    chunk = max(1, Budget // 2)
    while remaining > 0 and chunk >= 1:
        H = [e[i] + h_smart[i] for i in range(N)]
        cand_score = {}
        for (a, b, V) in storms:
            # search a margin beyond the rain window too: the ridge that
            # actually controls the outcome (a saddle a storm spills over)
            # can sit just outside where the rain itself falls
            margin = max(6, (b - a + 1) // 2)
            ae = max(0, a - margin)
            be = min(N - 1, b + margin)
            window_v = sum(v[a:b + 1])
            for idx in range(ae, be + 1):
                left_ok = (idx == 0) or (H[idx] >= H[idx - 1])
                right_ok = (idx == N - 1) or (H[idx] >= H[idx + 1])
                if left_ok and right_ok:
                    # weight by the value at stake in this storm, not the
                    # ridge's own height -- a low, cheap saddle guarding a
                    # lot of value matters more than a tall, irrelevant wall
                    cand_score[idx] = cand_score.get(idx, 0) + window_v
            mn_idx = a
            mn_val = H[a]
            for idx in range(a + 1, b + 1):
                if H[idx] < mn_val:
                    mn_val = H[idx]
                    mn_idx = idx
            cand_score[mn_idx] = cand_score.get(mn_idx, 0) + window_v
        cand = list(cand_score.keys())
        if len(cand) > MAX_CAND:
            cand.sort(key=lambda i: cand_score[i], reverse=True)
            cand = cand[:MAX_CAND]

        step = min(chunk, remaining)
        base_obj = objective(H, v, storms, N)
        best_gain = 0.0
        best_j = None
        for j in cand:
            old = H[j]
            H[j] = old + step
            val = objective(H, v, storms, N)
            H[j] = old
            gain = base_obj - val
            if gain > best_gain:
                best_gain = gain
                best_j = j

        if best_j is not None:
            h_smart[best_j] += step
            remaining -= step
        else:
            chunk = chunk // 2

    if remaining > 0:
        greedy_fill(e, v, storms, N, h_smart, remaining)

    # --- keep whichever allocation is actually better ---
    Hg = [e[i] + h_greedy[i] for i in range(N)]
    Hs = [e[i] + h_smart[i] for i in range(N)]
    Fg = objective(Hg, v, storms, N)
    Fs = objective(Hs, v, storms, N)
    h_final = h_smart if Fs <= Fg else h_greedy

    print(" ".join(map(str, h_final)))


if __name__ == "__main__":
    main()
