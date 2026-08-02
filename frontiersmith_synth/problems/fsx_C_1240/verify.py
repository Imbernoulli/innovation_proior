#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic scorer for the shard-key
assignment problem. Prints 'Ratio: <float in [0,1]>' on its own final line.
Minimization objective: lower total cost F is better.
"""
import sys
import math


def fail(msg):
    print("INFEASIBLE: %s" % msg)
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = toks[pos]
        pos += 1
        return v

    n = int(nxt())
    K = int(nxt())
    A = float(nxt())
    B = float(nxt())
    G = float(nxt())
    weights = [int(nxt()) for _ in range(n)]
    prev = [int(nxt()) for _ in range(n)]
    m = int(nxt())
    edges = []
    for _ in range(m):
        u = int(nxt())
        v = int(nxt())
        c = float(nxt())
        edges.append((u, v, c))
    return n, K, A, B, G, weights, prev, edges


def total_cost(assign, n, K, A, B, G, weights, prev, edges):
    loads = [0.0] * K
    for i in range(n):
        loads[assign[i]] += weights[i]
    avg = sum(weights) / K
    imbalance = sum((ld - avg) ** 2 for ld in loads)

    cross = 0.0
    for u, v, c in edges:
        if assign[u] != assign[v]:
            cross += c

    migration = 0.0
    for i in range(n):
        if prev[i] != -1 and assign[i] != prev[i]:
            migration += weights[i]

    return A * imbalance + B * cross + G * migration


def parse_output(path, n, K):
    try:
        with open(path) as f:
            toks = f.read().split()
    except Exception as e:
        fail("cannot read output: %s" % e)

    if len(toks) != n:
        fail("expected exactly %d shard assignments, got %d tokens" % (n, len(toks)))

    assign = []
    for i, tok in enumerate(toks):
        try:
            v = int(tok)
        except ValueError:
            fail("token %d (%r) is not an integer" % (i, tok))
            return
        if not (0 <= v < K):
            fail("shard index %d out of range [0,%d) at position %d" % (v, K, i))
        assign.append(v)
    return assign


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    n, K, A, B, G, weights, prev, edges = read_instance(inp)
    assign = parse_output(outp, n, K)

    F = total_cost(assign, n, K, A, B, G, weights, prev, edges)
    if not math.isfinite(F) or F < 0:
        fail("non-finite or negative total cost")

    # Internal reference baseline B_ref: round-robin key assignment
    # (assign[i] = i % K) -- the single most uniformly load-distributing
    # scheme possible, ignoring the transaction graph and prior epoch
    # entirely. It minimizes size skew but has no notion of co-location.
    rr = [i % K for i in range(n)]
    B_ref = total_cost(rr, n, K, A, B, G, weights, prev, edges)

    sc = min(1000.0, 100.0 * B_ref / max(1e-9, F))
    print("F=%.6f B_ref=%.6f" % (F, B_ref))
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
