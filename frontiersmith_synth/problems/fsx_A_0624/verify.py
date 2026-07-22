# Deterministic checker for "quota-chain-schoolmatch" (format C, maximize).
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Prints "... Ratio: <r>" with r in [0,1].
import sys, heapq

# Rank-1..rank-5 guild-point weights (steep drop-off; index = position in an
# apprentice's OWN preference list, which may be shorter than len(W)).
W = [100, 45, 8, 3, 1]


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def read_instance(path):
    toks = open(path).read().split()
    it = iter(toks)
    def nxt():
        return next(it)
    N = int(nxt()); M = int(nxt()); T = int(nxt())
    cap_max = [int(nxt()) for _ in range(M)]
    prefs = []
    for _ in range(N):
        Li = int(nxt())
        prefs.append([int(nxt()) for _ in range(Li)])
    order = [[int(nxt()) for _ in range(N)] for _ in range(M)]
    return N, M, T, cap_max, prefs, order


def run_da(N, M, prefs, prio_rank, caps):
    """Apprentice-proposing deferred acceptance. Returns match[i] (workshop id or -1).
    Each workshop keeps its top-caps[w] proposers by priority (smaller rank = higher)."""
    next_choice = [0] * N
    held = [[] for _ in range(M)]  # max-heap on priority rank via (-rank, apprentice)
    match = [-1] * N
    stack = list(range(N))
    while stack:
        s = stack.pop()
        pl = prefs[s]
        nc = next_choice[s]
        Ls = len(pl)
        while nc < Ls:
            wk = pl[nc]
            nc += 1
            r = prio_rank[wk][s]
            h = held[wk]
            if caps[wk] <= 0:
                continue  # zero-seat workshop: always rejects, try next choice
            if len(h) < caps[wk]:
                heapq.heappush(h, (-r, s))
                match[s] = wk
                break
            worst_negr, worst_s = h[0]
            if r < -worst_negr:
                heapq.heapreplace(h, (-r, s))
                match[s] = wk
                match[worst_s] = -1
                next_choice[worst_s] = _pos_after(prefs[worst_s], wk)
                stack.append(worst_s)
                break
            # else rejected by wk, keep advancing to the next listed choice
        next_choice[s] = nc
    return match


def _pos_after(pl, wk):
    for idx in range(len(pl)):
        if pl[idx] == wk:
            return idx + 1
    return len(pl)


def objective(N, M, prefs, prio_rank, caps):
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
    return U


def main():
    N, M, T, cap_max, prefs, order = read_instance(sys.argv[1])

    prio_rank = [[0] * N for _ in range(M)]
    for wk in range(M):
        ow = order[wk]
        pr = prio_rank[wk]
        for pos in range(N):
            pr[ow[pos]] = pos

    # ---- internal baseline B: uniform capacities (as even as possible) ----
    uniform = [T // M] * M
    for j in range(T % M):
        uniform[j] += 1
    B = objective(N, M, prefs, prio_rank, uniform)
    B = max(1, B)

    # ---- parse + validate participant capacities ----
    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")
    if len(otoks) != M:
        fail("need exactly %d integers, got %d" % (M, len(otoks)))
    caps = []
    for t in otoks:
        try:
            v = int(t)
        except Exception:
            fail("non-integer/non-finite capacity %r" % t)
        caps.append(v)
    for j, c in enumerate(caps):
        if c < 0:
            fail("negative capacity at workshop %d" % j)
        if c > cap_max[j]:
            fail("capacity %d at workshop %d exceeds room limit %d" % (c, j, cap_max[j]))
    if sum(caps) != T:
        fail("capacities sum to %d, must equal T=%d" % (sum(caps), T))

    F = objective(N, M, prefs, prio_rank, caps)
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
