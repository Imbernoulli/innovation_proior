import sys

# Format D checker -- shared pivot order across K sparse symmetric patterns.
#
# Input <in>:
#   N K
#   M_1
#   u1 v1 u2 v2 ... (M_1 pairs, pattern 1 edges, 1-indexed)
#   M_2
#   ... (pattern 2 edges)
#   ... (K patterns total)
#
# Participant <out>: a single permutation of 1..N (whitespace separated),
# the ONE elimination (pivot) order to be used for ALL K patterns.
#
# 1) EXACT feasibility gate: output must be a bijection on {1..N} (parse ints
#    strictly; non-finite/garbage/duplicate/out-of-range -> Ratio 0.0).
# 2) For each pattern, run the exact symbolic elimination game (fill-in) under
#    the submitted order and sum the scalar-multiply op count. Total F = sum
#    over the K patterns.
# 3) Baseline B = same elimination game, summed over the K patterns, under
#    the IDENTITY order 1,2,...,N -- a trivial, structure-blind construction
#    the checker builds itself (NOT the intended answer).
# 4) Objective is MINIMIZE: Ratio = min(1000, 100*B/max(1e-9,F)) / 1000.


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def fillin_ops(N, edge_list, order):
    """Exact symbolic elimination game via N-bit adjacency masks.
    order: list of vertices (1-indexed), first eliminated first.
    Returns total scalar-multiply op count (int)."""
    adj = [0] * (N + 1)
    for (u, v) in edge_list:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    ops = 0
    for v in order:
        nbrs_mask = adj[v]
        d = bin(nbrs_mask).count("1")
        ops += d * (d + 1) // 2
        m = nbrs_mask
        while m:
            low = m & (-m)
            u = low.bit_length() - 1
            adj[u] |= nbrs_mask
            adj[u] &= ~low            # drop self bit
            adj[u] &= ~(1 << v)       # drop the vertex being eliminated
            m &= m - 1
        adj[v] = 0
    return ops


def main():
    try:
        inp = open(sys.argv[1]).read().split()
    except Exception:
        fail("cannot read input")
    it = iter(inp)

    def nxt_int():
        return int(next(it))

    try:
        N = nxt_int()
        K = nxt_int()
    except Exception:
        fail("bad header")
    if not (4 <= N <= 5000 and 1 <= K <= 64):
        fail("bad N/K")

    patterns = []
    try:
        for _k in range(K):
            M = nxt_int()
            if M < 0 or M > 2_000_000:
                fail("bad M")
            edges = []
            for _e in range(M):
                u = nxt_int()
                v = nxt_int()
                if not (1 <= u <= N and 1 <= v <= N) or u == v:
                    fail("bad edge in instance")
                edges.append((u, v) if u < v else (v, u))
            patterns.append(edges)
    except (StopIteration, ValueError):
        fail("truncated/corrupt instance")

    # ---- parse participant output ----
    try:
        out_toks = open(sys.argv[2]).read().split()
    except Exception:
        fail("cannot read output")
    if len(out_toks) != N:
        fail("wrong token count (got %d, need %d)" % (len(out_toks), N))
    order = []
    seen = [False] * (N + 1)
    for tok in out_toks:
        try:
            x = int(tok)
        except Exception:
            fail("non-integer/non-finite token %r" % tok)
        if not (1 <= x <= N):
            fail("token out of range: %d" % x)
        if seen[x]:
            fail("duplicate vertex %d (not a permutation)" % x)
        seen[x] = True
        order.append(x)
    if len(order) != N or not all(seen[1:N + 1]):
        fail("not a bijection on 1..N")

    # ---- objective F: sum of fill-in op counts over all K patterns ----
    F = 0
    for edges in patterns:
        F += fillin_ops(N, edges, order)

    # ---- baseline B: identity order 1..N, same metric ----
    base_order = list(range(1, N + 1))
    B = 0
    for edges in patterns:
        B += fillin_ops(N, edges, base_order)

    if B <= 0:
        fail("degenerate baseline")

    sc = min(1000.0, 100.0 * B / max(1e-9, float(F)))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
