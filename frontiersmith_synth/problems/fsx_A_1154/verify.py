#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>  -- checker for fsx_A_1154 (Spectrum-Preserving Sparsifier Duel).

Scores a candidate reweighted sub-selection H (<= s edges of G, each kept weight in
[0, original_weight]) by the WORST relative error, over the PUBLISHED test-vector family
x^(1)..x^(K) (given in the input), of the quadratic form x^T L x:

    F(H) = max_k | (x_k^T L_H x_k) / (x_k^T L_G x_k) - 1 |     (minimize)

Internal baseline B: keep the s heaviest original edges (unchanged weight) -- a naive,
test-vector-blind construction. Ratio = min(1000, 100*B/F) / 1000.
"""
import sys
import math


def fail(msg):
    print(f"INFEASIBLE: {msg} Ratio: 0.0")
    sys.exit(0)


def read_tokens(path):
    with open(path, "r") as f:
        return f.read().split()


def main():
    if len(sys.argv) < 3:
        fail("usage")
    in_path, out_path = sys.argv[1], sys.argv[2]

    itoks = read_tokens(in_path)
    if len(itoks) < 4:
        fail("malformed input")
    try:
        n = int(itoks[0]); m = int(itoks[1]); s = int(itoks[2]); K = int(itoks[3])
    except ValueError:
        fail("malformed input header")
    if n <= 0 or m < 0 or s < 0 or K <= 0:
        fail("bad input header")

    pos = 4
    edges = []  # (u, v, w) 1-indexed
    edge_index = {}  # (min,max) -> (weight, first_seen_index)
    for ei in range(m):
        if pos + 3 > len(itoks):
            fail("truncated input edges")
        u = int(itoks[pos]); v = int(itoks[pos + 1]); w = float(itoks[pos + 2]); pos += 3
        edges.append((u, v, w))
        key = (min(u, v), max(u, v))
        if key not in edge_index:
            edge_index[key] = (w, ei)

    test_vecs = []
    for _k in range(K):
        if pos + n > len(itoks):
            fail("truncated input test vectors")
        x = [float(t) for t in itoks[pos:pos + n]]
        pos += n
        test_vecs.append(x)

    def quad_form(edge_list, xs):
        """xs: list of K vectors. returns list of K quadratic-form values over edge_list."""
        totals = [0.0] * len(xs)
        for (u, v, w) in edge_list:
            du = u - 1
            dv = v - 1
            for ki, x in enumerate(xs):
                diff = x[du] - x[dv]
                totals[ki] += w * diff * diff
        return totals

    t_targets = quad_form(edges, test_vecs)
    for tk in t_targets:
        if not (tk > 0.0):
            fail("degenerate instance: a published test vector has zero energy on G")

    # ---- internal baseline B: s heaviest original edges, unchanged weight ----
    order = sorted(range(m), key=lambda i: -edges[i][2])  # stable: ties by original index
    base_sel = [edges[i] for i in order[:s]]
    t_base = quad_form(base_sel, test_vecs)
    F_base = max(abs(t_base[k] / t_targets[k] - 1.0) for k in range(K))
    B = F_base
    if not (B > 0.0):
        B = 1e-9

    # ---- parse candidate output ----
    otoks = read_tokens(out_path)
    if len(otoks) < 1:
        fail("empty output")
    try:
        k_count = int(otoks[0])
    except ValueError:
        fail("bad edge count token")
    if k_count < 0 or k_count > s:
        fail(f"edge count {k_count} out of [0,{s}]")
    need = 1 + 3 * k_count
    if len(otoks) != need:
        fail(f"expected exactly {need} tokens, got {len(otoks)}")

    sel_edges = []
    seen_keys = set()
    p = 1
    for _i in range(k_count):
        tu, tv, tw = otoks[p], otoks[p + 1], otoks[p + 2]
        p += 3
        try:
            u = int(tu); v = int(tv); w = float(tw)
        except ValueError:
            fail("non-numeric edge token")
        if not math.isfinite(w):
            fail("non-finite weight")
        if u < 1 or u > n or v < 1 or v > n or u == v:
            fail("edge endpoint out of range / self-loop")
        if w < -1e-9:
            fail("negative weight")
        w = max(0.0, w)
        key = (min(u, v), max(u, v))
        if key not in edge_index:
            fail(f"edge ({u},{v}) is not part of the input graph G")
        if key in seen_keys:
            fail(f"duplicate edge ({u},{v}) in output")
        seen_keys.add(key)
        orig_w, _idx = edge_index[key]
        if w > orig_w + 1e-6:
            fail(f"edge ({u},{v}) reweighted above its original weight (dim-only allowed)")
        sel_edges.append((u, v, w))

    t_out = quad_form(sel_edges, test_vecs)
    F = max(abs(t_out[k] / t_targets[k] - 1.0) for k in range(K))

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("B=%.6f F=%.6f Ratio: %.6f" % (B, F, sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
