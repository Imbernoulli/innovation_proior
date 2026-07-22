# TIER: strong
"""The insight: latency is endogenous. A "ready" load's true cost (hit or
miss) is not a property of the load -- it is a property of what the
schedule itself has recently done to that cache set. A fixed-latency list
scheduler cannot see this because it never asks "given what I have already
committed to, would this address still be resident right now?".

So: instead of committing to a static priority order up front, replay the
SAME hazard+cache machine the checker uses while constructing the
schedule. At every cycle, among all ops that are truly ready *right now*
(dependencies already issued and their results already available, exactly
the checker's own readiness rule), prefer a load/store whose address is
currently cache-resident (a guaranteed hit) over one that would evict and
miss; among equal-quality options, prefer continuing whichever chain was
just active (keeps a chain's own line warm) and otherwise favor plain ALU
work, which carries no cache risk at all. A load that would miss right now
is only taken when nothing safer is ready -- often that means it has, by
construction, waited long enough for a colliding chain to finish, turning
what a naive scheduler treats as an unavoidable miss into a hit."""
import sys

ALU_LAT = 2
LOAD_HIT_LAT = 3
LOAD_MISS_LAT = 15
STORE_LAT = 1


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)

    def nx():
        return next(it)

    N = int(nx())
    M = int(nx())
    S = int(nx())

    ntype = [None] * N
    naddr = [None] * N
    for _ in range(N):
        oid = int(nx())
        t = nx()
        if t == "A":
            ntype[oid] = "A"
        else:
            ntype[oid] = t
            naddr[oid] = int(nx())

    preds = [[] for _ in range(N)]
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u = int(nx())
        v = int(nx())
        preds[v].append(u)
        adj[u].append(v)

    parent = list(range(N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for u in range(N):
        for v in adj[u]:
            union(u, v)

    indeg = [len(preds[i]) for i in range(N)]
    unlocked = set(i for i in range(N) if indeg[i] == 0)
    avail = [None] * N
    cache = {}
    order = []
    cycle = 0
    last_comp = None

    def issue(i, cur_cycle):
        if ntype[i] == "A":
            lat = ALU_LAT
        elif ntype[i] == "L":
            a = naddr[i]
            s = a % S
            hit = cache.get(s) == a
            cache[s] = a
            lat = LOAD_HIT_LAT if hit else LOAD_MISS_LAT
        else:
            a = naddr[i]
            s = a % S
            cache[s] = a
            lat = STORE_LAT
        avail[i] = cur_cycle + lat

    def pick(mem_blocked):
        best, best_key = None, None
        for i in unlocked:
            if mem_blocked and ntype[i] != "A":
                continue
            if not all(avail[p] <= cycle for p in preds[i]):
                continue
            if ntype[i] == "L":
                would_hit = cache.get(naddr[i] % S) == naddr[i]
                level = 0 if would_hit else 3
            else:
                level = 1
            affinity = 0 if find(i) == last_comp else 1
            key = (level, affinity, i)
            if best is None or key < best_key:
                best, best_key = i, key
        return best

    while len(order) < N:
        c0 = pick(mem_blocked=False)
        if c0 is None:
            cycle += 1
            continue
        issue(c0, cycle)
        unlocked.discard(c0)
        order.append(c0)
        last_comp = find(c0)
        for w in adj[c0]:
            indeg[w] -= 1
            if indeg[w] == 0:
                unlocked.add(w)
        slot0_mem = ntype[c0] != "A"
        c1 = pick(mem_blocked=slot0_mem)
        if c1 is not None:
            issue(c1, cycle)
            unlocked.discard(c1)
            order.append(c1)
            last_comp = find(c1)
            for w in adj[c1]:
                indeg[w] -= 1
                if indeg[w] == 0:
                    unlocked.add(w)
        cycle += 1

    sys.stdout.write(" ".join(map(str, order)))


if __name__ == "__main__":
    main()
