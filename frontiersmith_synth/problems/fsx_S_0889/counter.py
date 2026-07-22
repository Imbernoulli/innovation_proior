#!/usr/bin/env python3
"""
Checker for fsx_S_0889 (Hot-Loop Scheduler for an In-Order Dual-Issue Core).

Input (<in>):
  line 1:      N M S
  next N lines: "id A"  |  "id L addr"  |  "id S addr"
  next M lines: "u v"    (u must be issued strictly before v)

Participant output (<out>): N whitespace-separated integers -- a permutation
of 0..N-1, the chosen static issue order (the "hand schedule").

Machine model (fixed constants, same for every test):
  - 2-wide in-order issue, STRICT program order: at every cycle the pipeline
    only ever attempts to issue the next not-yet-issued instruction (the
    "head") into slot0, and -- only if slot0 issued -- the following
    instruction into slot1. Nothing may bypass a stalled head.
  - A node may issue only once every one of its predecessors' results is
    already available (>= at the START of the cycle); same-cycle
    (slot0->slot1) forwarding is not physically possible since every
    latency is >= 1 cycle.
  - Structural hazard: a single memory port. slot1 may not carry a LOAD or
    STORE if slot0 already carries a LOAD or STORE.
  - ALU: fixed latency ALU_LAT.
  - LOAD: latency depends on a direct-mapped cache of S sets (1 line per
    set, no offsets: set = addr % S). A hit costs LOAD_HIT_LAT, a miss
    costs LOAD_MISS_LAT; either way the touched set now holds this addr
    (evicting whatever else was resident, allocate-on-miss/refresh-on-hit).
  - STORE: fixed latency STORE_LAT (nothing waits on a store's "value") but
    STILL touches the cache the same way a load does, so it can silently
    evict another chain's resident line.

Score (minimization): F = cycle count of the submitted order. The checker
also runs the SAME simulator over its own cache-blind, ascending-id
Kahn-topological-sort baseline order to get B. Ratio = min(1, 0.1*B/F):
matching the baseline scores 0.1, and a 10x-lower cycle count than the
baseline already saturates the score at 1.0.
"""
import sys
import heapq

ALU_LAT = 2
LOAD_HIT_LAT = 3
LOAD_MISS_LAT = 15
STORE_LAT = 1


def fail(reason):
    print("INVALID:", reason)
    print("Ratio: 0.0")
    sys.exit(0)


def read_input(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)

    def nx():
        return next(it)

    N = int(nx())
    M = int(nx())
    S = int(nx())
    ntype = [None] * N
    naddr = [None] * N
    seen_ids = set()
    for _ in range(N):
        oid = int(nx())
        if not (0 <= oid < N) or oid in seen_ids:
            raise ValueError("bad node id in input")
        seen_ids.add(oid)
        t = nx()
        if t == "A":
            ntype[oid] = "A"
            naddr[oid] = None
        else:
            ntype[oid] = t
            naddr[oid] = int(nx())
    preds = [[] for _ in range(N)]
    adj = [[] for _ in range(N)]
    edges = []
    for _ in range(M):
        u = int(nx())
        v = int(nx())
        edges.append((u, v))
        preds[v].append(u)
        adj[u].append(v)
    return N, M, S, ntype, naddr, preds, adj, edges


def simulate(order, ntype, naddr, preds, S):
    N = len(order)
    avail = [-1] * N
    cache = {}
    cycle = 0
    head = 0

    def issue(i, cur_cycle):
        if ntype[i] == "A":
            lat = ALU_LAT
        elif ntype[i] == "L":
            a = naddr[i]
            s = a % S
            hit = cache.get(s) == a
            cache[s] = a
            lat = LOAD_HIT_LAT if hit else LOAD_MISS_LAT
        else:  # store
            a = naddr[i]
            s = a % S
            cache[s] = a
            lat = STORE_LAT
        avail[i] = cur_cycle + lat

    while head < N:
        i = order[head]
        ready_i = all(avail[p] <= cycle for p in preds[i])
        if not ready_i:
            cycle += 1
            continue
        issue(i, cycle)
        slot0_mem = ntype[i] != "A"
        head += 1
        if head < N:
            j = order[head]
            j_mem = ntype[j] != "A"
            if not (slot0_mem and j_mem):
                ready_j = all(avail[p] <= cycle for p in preds[j])
                if ready_j:
                    issue(j, cycle)
                    head += 1
        cycle += 1
    return cycle


def kahn_baseline(N, adj, preds):
    indeg = [len(preds[i]) for i in range(N)]
    ready = [i for i in range(N) if indeg[i] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        o = heapq.heappop(ready)
        order.append(o)
        for w in adj[o]:
            indeg[w] -= 1
            if indeg[w] == 0:
                heapq.heappush(ready, w)
    return order


def main():
    if len(sys.argv) < 3:
        fail("usage: counter.py <in> <out> [<ans>]")
    in_path, out_path = sys.argv[1], sys.argv[2]

    try:
        N, M, S, ntype, naddr, preds, adj, edges = read_input(in_path)
    except Exception as e:
        fail(f"malformed input: {e}")

    try:
        with open(out_path) as f:
            out_toks = f.read().split()
    except Exception as e:
        fail(f"cannot read output: {e}")

    if len(out_toks) != N:
        fail(f"expected exactly {N} tokens, got {len(out_toks)}")

    order = []
    seen = [False] * N
    for tok in out_toks:
        try:
            v = int(tok)
        except Exception:
            fail(f"non-integer token: {tok!r}")
        if not (0 <= v < N):
            fail(f"token out of range: {v}")
        if seen[v]:
            fail(f"duplicate id in output: {v}")
        seen[v] = True
        order.append(v)

    pos = [0] * N
    for idx, v in enumerate(order):
        pos[v] = idx
    for u, v in edges:
        if pos[u] >= pos[v]:
            fail(f"dependence violated: {u} must precede {v}")

    F = simulate(order, ntype, naddr, preds, S)
    if F <= 0:
        fail("non-positive simulated cycle count")

    base_order = kahn_baseline(N, adj, preds)
    B = simulate(base_order, ntype, naddr, preds, S)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print(f"N={N} M={M} S={S} F={F} B={B}")
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
