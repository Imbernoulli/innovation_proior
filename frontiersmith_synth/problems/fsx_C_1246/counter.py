#!/usr/bin/env python3
"""counter.py <in> <out> <ans> -- deterministic scorer for ann-index-build.

<in>  : N M R, then N dataset points, then Q, then Q query points (all ints).
<out> : the participant artifact --
            R entry-point indices (distinct, in [0,N))
            N adjacency lines, each "deg n_1 .. n_deg" with 0<=deg<=M
<ans> : ignored placeholder.

The judge re-runs a FIXED greedy best-first search (never anything the
participant controls) starting simultaneously from all R declared entry
points, for every held-out query, and counts distance evaluations. A query
whose search does not land on the true nearest neighbour (recomputed here
by brute force) is charged a full linear-scan penalty of N -- representing
"fell back to exhaustive scan to guarantee the answer". Everything is exact
integer arithmetic (squared Euclidean distance): fully deterministic.
"""
import sys


def die0(reason):
    sys.stderr.write(f"INFEASIBLE: {reason}\n")
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        die0("bad invocation")
    inf, outf = sys.argv[1], sys.argv[2]

    with open(inf) as f:
        itoks = f.read().split()
    ii = iter(itoks)
    N = int(next(ii)); M = int(next(ii)); R = int(next(ii))
    pts = []
    for _ in range(N):
        x = int(next(ii)); y = int(next(ii))
        pts.append((x, y))
    Q = int(next(ii))
    queries = []
    for _ in range(Q):
        x = int(next(ii)); y = int(next(ii))
        queries.append((x, y))

    try:
        raw = open(outf).read()
    except Exception as e:
        die0(f"cannot read output: {e}")
    otoks = raw.split()
    # bound the number of tokens we will ever try to parse, so a huge/garbage
    # file cannot make the checker do unbounded work -- we need at most
    # R + N + sum(deg) <= R + N + N*M tokens for a feasible artifact anyway.
    max_toks = R + N + N * (M + 1) + 8
    if len(otoks) > max_toks:
        die0(f"output has {len(otoks)} tokens, more than the {max_toks} a feasible artifact could use")

    oi = iter(otoks)

    def onext():
        return next(oi, None)

    def parse_int(tok, what):
        if tok is None:
            die0(f"missing token: {what}")
        try:
            if tok.lower() in ("nan", "inf", "-inf", "+inf", "infinity", "-infinity"):
                raise ValueError
            v = int(tok)
        except ValueError:
            die0(f"non-integer / non-finite token for {what}: {tok!r}")
        return v

    # ---- entry points ----
    entries = []
    for k in range(R):
        v = parse_int(onext(), f"entry[{k}]")
        if not (0 <= v < N):
            die0(f"entry[{k}]={v} out of range [0,{N})")
        entries.append(v)
    if len(set(entries)) != R:
        die0("duplicate entry-point indices")

    # ---- adjacency ----
    adj = []
    for i in range(N):
        deg = parse_int(onext(), f"deg[{i}]")
        if not (0 <= deg <= M):
            die0(f"node {i} degree {deg} outside [0,{M}]")
        nbrs = []
        for k in range(deg):
            v = parse_int(onext(), f"nbr[{i}][{k}]")
            if not (0 <= v < N):
                die0(f"node {i} neighbour {v} out of range [0,{N})")
            if v == i:
                die0(f"node {i} has a self-loop")
            nbrs.append(v)
        if len(set(nbrs)) != len(nbrs):
            die0(f"node {i} has a duplicate neighbour")
        adj.append(nbrs)

    if onext() is not None:
        die0("trailing tokens after the expected artifact")

    # ---- brute-force ground truth per query ----
    def d2(a, b):
        dx = a[0] - b[0]; dy = a[1] - b[1]
        return dx * dx + dy * dy

    true_d = [min(d2(q, p) for p in pts) for q in queries]

    # ---- fixed greedy best-first search, counting distance evaluations ----
    hop_cap = 2 * N + 50  # safety net; the search is strictly monotonic so it
                          # cannot actually cycle (see statement), this only
                          # guards against a pathological huge N
    total_cost = 0
    for qi, q in enumerate(queries):
        cost = 0
        best_node, best_d = None, None
        for e in entries:
            dd = d2(q, pts[e])
            cost += 1
            if best_d is None or dd < best_d:
                best_d, best_node = dd, e
        hops = 0
        improved = True
        while improved and hops < hop_cap:
            improved = False
            hops += 1
            cand_d, cand_node = best_d, best_node
            for nb in adj[best_node]:
                dd = d2(q, pts[nb])
                cost += 1
                if dd < cand_d:
                    cand_d, cand_node = dd, nb
                    improved = True
            if improved:
                best_d, best_node = cand_d, cand_node
        if best_d == true_d[qi]:
            total_cost += cost
        else:
            total_cost += N  # recall miss -> pay for a full linear scan

    F = total_cost / max(1, Q)
    B = float(N)  # trivial feasible baseline: literally scan all N points per query
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F(avg distance-evals/query)=%.4f  B(brute-force)=%.4f  Q=%d" % (F, B, Q))
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
