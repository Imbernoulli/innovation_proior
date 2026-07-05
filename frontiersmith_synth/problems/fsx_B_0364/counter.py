#!/usr/bin/env python3
# Format D checker (eval_form=flops / op-count) -- deep-sea cable-swap routing.
#
# CLI:  python3 counter.py <in> <out> <ans>   (ans ignored)
#
#   1) Parse the instance from <in>: mesh graph, home placement, ordered splice schedule.
#   2) Parse the participant's move list from <out>.  Each move is either:
#          S u v   -- swap the channels on boxes u and v; (u,v) must be a cable.
#          G i     -- execute splice #i (1-based); i must be the NEXT pending splice and
#                     its two channels must currently sit on adjacent (cabled) boxes.
#      Track the live channel<->box mapping.  This EXACTLY certifies functional
#      equivalence: every splice is applied to its true channel pair, in order, at
#      adjacency -- so the routed circuit realises the identical logical schedule.
#      Any illegal swap, out-of-order/non-adjacent splice, missing splice, bad token,
#      or non-finite value  ->  Ratio: 0.0.
#   3) Objective (MINIMISE) = number of SWAP moves F.
#   4) Internal baseline B (checker builds it itself): the naive "reset-to-home" router --
#      for each splice, route the first channel to its partner along a shortest cable path
#      (d-1 swaps) and then UNDO those swaps to restore the home placement (d-1 swaps),
#      i.e. cost 2*(d-1) per splice with d = home distance.  B = sum over splices.
#   5) Ratio = min(1, 0.1 * B / F)   (minimisation; trivial reset-router scores 0.1).

import sys
from collections import deque

MAX_TOKENS = 20_000_000


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def bfs_dist(adj, src, dst):
    if src == dst:
        return 0
    n = len(adj)
    dist = [-1] * n
    dist[src] = 0
    dq = deque([src])
    while dq:
        x = dq.popleft()
        for y in adj[x]:
            if dist[y] < 0:
                dist[y] = dist[x] + 1
                if y == dst:
                    return dist[y]
                dq.append(y)
    return -1  # disconnected (should not happen on a mesh)


def main():
    try:
        inp = open(sys.argv[1]).read().split()
    except Exception:
        fail("cannot read instance")
    try:
        raw = open(sys.argv[2]).read()
    except Exception:
        fail("cannot read output")

    # ---- parse instance ----
    it = iter(inp)
    try:
        n = int(next(it)); e = int(next(it)); q = int(next(it))
    except Exception:
        fail("bad instance header")
    if n <= 0 or e < 0 or q <= 0:
        fail("bad instance dims")

    adj = [[] for _ in range(n)]
    edgeset = set()
    try:
        for _ in range(e):
            u = int(next(it)); v = int(next(it))
            adj[u].append(v)
            adj[v].append(u)
            edgeset.add((u, v) if u < v else (v, u))
    except Exception:
        fail("bad edges")

    try:
        placement = [int(next(it)) for _ in range(n)]
    except Exception:
        fail("bad placement")
    if sorted(placement) != list(range(n)):
        fail("placement not a permutation")

    ops = []
    try:
        for _ in range(q):
            a = int(next(it)); b = int(next(it))
            ops.append((a, b))
    except Exception:
        fail("bad ops")

    # ---- internal baseline B (reset-to-home router) ----
    home_site = [0] * n
    for s, tok in enumerate(placement):
        home_site[tok] = s
    B = 0
    for (a, b) in ops:
        d = bfs_dist(adj, home_site[a], home_site[b])
        if d < 0:
            fail("instance disconnected")
        B += 2 * (d - 1)
    if B <= 0:
        fail("degenerate instance (baseline 0)")

    # ---- parse + simulate participant output ----
    # reject non-finite floats explicitly (harness feeds nan/inf floods)
    low = raw.lower()
    if "nan" in low or "inf" in low:
        fail("non-finite token")
    toks = raw.split()
    if len(toks) > MAX_TOKENS:
        fail("output too large")

    token_at = list(placement)          # token_at[site] = token
    site_at = [0] * n                   # site_at[token] = site
    for s, tok in enumerate(placement):
        site_at[tok] = s

    adjset_local = edgeset  # (min,max)

    swap_count = 0
    next_op = 0   # 0-based index of next splice to execute
    i = 0
    L = len(toks)
    while i < L:
        cmd = toks[i]; i += 1
        if cmd == "S":
            if i + 1 >= L:
                fail("truncated swap")
            try:
                u = int(toks[i]); v = int(toks[i + 1])
            except Exception:
                fail("bad swap ints")
            i += 2
            if not (0 <= u < n and 0 <= v < n) or u == v:
                fail("swap out of range")
            key = (u, v) if u < v else (v, u)
            if key not in adjset_local:
                fail("swap on non-cable")
            tu = token_at[u]; tv = token_at[v]
            token_at[u] = tv; token_at[v] = tu
            site_at[tu] = v; site_at[tv] = u
            swap_count += 1
        elif cmd == "G":
            if i >= L:
                fail("truncated gate")
            try:
                idx = int(toks[i])
            except Exception:
                fail("bad gate index")
            i += 1
            if idx != next_op + 1:
                fail("splice out of order (got %d, expected %d)" % (idx, next_op + 1))
            a, b = ops[next_op]
            su = site_at[a]; sv = site_at[b]
            key = (su, sv) if su < sv else (sv, su)
            if key not in adjset_local:
                fail("splice #%d channels not on adjacent boxes" % idx)
            next_op += 1
        else:
            fail("unknown command token")

    if next_op != q:
        fail("not all splices executed (%d/%d)" % (next_op, q))

    F = swap_count
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("baseline_B=%d yours_F=%d Ratio: %.6f" % (B, F, sc / 1000.0))


if __name__ == "__main__":
    main()
