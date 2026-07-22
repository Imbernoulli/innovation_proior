#!/usr/bin/env python3
"""counter.py <in> <out> <ans> -- deterministic FLOPs checker for "Shared Stockpot Chain".

The participant artifact declares a DAG of combine-nodes over the base potion chain
(leaves = base potions, split-nodes = pairwise combine of two ADJACENT already-built
sub-ranges). A node's range is *derived* bottom-up from its children (never taken on
faith), so a submission cannot lie about what range it has built. For every brew order
(query) the checker requires some declared node whose derived range equals exactly the
query's [L,R). The scored cost is the sum of the combine-op cost (d_l*d_k*d_r) of every
DISTINCT node id reachable from the union of all query roots -- i.e. a node shared by
several queries is paid for exactly once, however many queries reach it.
"""
import sys


def fail(reason):
    sys.stderr.write("INFEASIBLE: %s\n" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0")
        return
    inf, outf = sys.argv[1], sys.argv[2]

    # ---- read the instance (trusted: produced by gen.py) ----
    with open(inf) as f:
        in_toks = f.read().split()
    p = 0
    m = int(in_toks[p]); p += 1
    dims = [int(in_toks[p + i]) for i in range(m + 1)]; p += m + 1
    Q = int(in_toks[p]); p += 1
    queries = []
    for _ in range(Q):
        L = int(in_toks[p]); R = int(in_toks[p + 1]); p += 2
        queries.append((L, R))

    # ---- read the participant artifact (untrusted) ----
    try:
        with open(outf) as f:
            out_toks = f.read().split()
    except Exception:
        fail("cannot read output file")

    if not out_toks:
        fail("empty output")

    MAX_NODES = 400000
    pos = 0

    def next_tok():
        nonlocal pos
        if pos >= len(out_toks):
            raise ValueError("unexpected end of output")
        t = out_toks[pos]
        pos += 1
        return t

    def next_int():
        t = next_tok()
        tl = t.lower()
        if tl in ("nan", "inf", "+inf", "-inf", "infinity", "-infinity"):
            raise ValueError("non-finite token %r" % t)
        try:
            if t.lstrip("+-").isdigit():
                return int(t)
            raise ValueError
        except ValueError:
            raise ValueError("expected an integer token, got %r" % t)

    try:
        N = next_int()
        if N < 1 or N > MAX_NODES:
            raise ValueError("node count %d out of allowed range [%d,%d]" % (N, 1, MAX_NODES))

        node_lo = [0] * N
        node_hi = [0] * N
        node_kids = [None] * N  # None = leaf, else (c1, c2)

        for nid in range(N):
            tag = next_tok()
            if tag == "L":
                i = next_int()
                if not (0 <= i < m):
                    raise ValueError("leaf index %d out of range at node %d" % (i, nid))
                node_lo[nid] = i
                node_hi[nid] = i + 1
            elif tag == "S":
                c1 = next_int()
                c2 = next_int()
                if not (0 <= c1 < nid):
                    raise ValueError("child id %d must reference an earlier node (node %d)" % (c1, nid))
                if not (0 <= c2 < nid):
                    raise ValueError("child id %d must reference an earlier node (node %d)" % (c2, nid))
                if c1 == c2:
                    raise ValueError("split node %d has two identical children" % nid)
                if node_hi[c1] != node_lo[c2]:
                    raise ValueError(
                        "node %d combines non-adjacent ranges [%d,%d) and [%d,%d)"
                        % (nid, node_lo[c1], node_hi[c1], node_lo[c2], node_hi[c2])
                    )
                node_lo[nid] = node_lo[c1]
                node_hi[nid] = node_hi[c2]
                node_kids[nid] = (c1, c2)
            else:
                raise ValueError("unknown node tag %r at node %d (want L or S)" % (tag, nid))

        roots = [next_int() for _ in range(Q)]
        if pos != len(out_toks):
            raise ValueError("trailing garbage after the %d roots" % Q)

        for qi, rid in enumerate(roots):
            if not (0 <= rid < N):
                raise ValueError("root for query %d references invalid node id %d" % (qi, rid))
            L, R = queries[qi]
            if node_lo[rid] != L or node_hi[rid] != R:
                raise ValueError(
                    "query %d wants range [%d,%d) but declared root %d has range [%d,%d)"
                    % (qi, L, R, rid, node_lo[rid], node_hi[rid])
                )
    except ValueError as e:
        fail(str(e))
    except Exception as e:  # noqa: BLE001 - any parse anomaly is an infeasible submission
        fail("parse error: %s" % e)

    # ---- reachability: union of everything needed by the Q roots ----
    reachable = bytearray(N)
    stack = []
    for r in set(roots):
        if not reachable[r]:
            reachable[r] = 1
            stack.append(r)
    while stack:
        nid = stack.pop()
        kids = node_kids[nid]
        if kids is not None:
            for c in kids:
                if not reachable[c]:
                    reachable[c] = 1
                    stack.append(c)

    F = 0
    for nid in range(N):
        if reachable[nid] and node_kids[nid] is not None:
            c1, c2 = node_kids[nid]
            a = node_lo[c1]
            k = node_hi[c1]
            b = node_hi[c2]
            F += dims[a] * dims[k] * dims[b]

    if F <= 0:
        fail("total combine cost is zero (impossible for a well-formed multi-potion query)")

    # ---- checker's own trivial baseline: independent left-to-right fold, NO sharing ----
    B = 0
    for (L, R) in queries:
        t = L + 1
        while t < R:
            B += dims[L] * dims[t] * dims[t + 1]
            t += 1

    reach_count = sum(reachable)
    # Sub-linear (concave) scaling in the reduction factor B/F, rather than the raw
    # linear ratio: with 50 heavily-overlapping brew orders a good shared plan can push
    # B/F past 10x purely from reuse multiplicity, which would saturate a linear
    # 100*B/F score at the 1.0 ceiling and erase headroom above the shipped `strong`
    # reference. The exponent 0.85 keeps trivial (B/F=1) pinned at exactly 0.1 while
    # damping large reduction factors so the true (unknown, possibly-better) joint
    # optimum still has visible room above every reference solution shipped here.
    ratio_bf = B / max(1e-9, F)
    sc = min(1000.0, 100.0 * (ratio_bf ** 0.85))
    sys.stderr.write(
        "m=%d Q=%d N=%d reachable=%d F=%d B=%d\n" % (m, Q, N, reach_count, F, B)
    )
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
