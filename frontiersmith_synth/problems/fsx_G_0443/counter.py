#!/usr/bin/env python3
"""Deterministic checker for tensor-network contraction ORDER (format D).

CLI: python3 counter.py <in> <out> <ans>   (ans ignored).

The participant submits a contraction schedule: a sequence of m-1 pairwise
merges.  Initial tensors have ids 0..m-1.  The i-th merge line (0-indexed)
consumes two currently-live tensor ids and produces a NEW tensor with id
m+i.  The schedule is valid iff it consumes every leaf and leaves exactly one
tensor.

Cost model (exact scalar-multiplication count, big-int arithmetic):
  merging tensors with live index sets A,B costs  prod_{i in A|B} dim[i]
  (one multiply per element of the combined index space).  Indices that no
  longer appear on any other live tensor and are not open legs are summed
  out of the result.

Score (minimization):  B = cost of a deterministic min-cost greedy schedule
(the checker builds it itself);  F = participant cost;
Ratio = min(1, 0.1 * B / F).
"""
import sys


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    m = int(next(it))
    k = int(next(it))
    dims = [int(next(it)) for _ in range(k)]
    tensors = []
    for _ in range(m):
        deg = int(next(it))
        tensors.append([int(next(it)) for _ in range(deg)])
    # open leg = index that appears on exactly one tensor
    occ = [0] * k
    for t in tensors:
        for i in t:
            occ[i] += 1
    output = [c == 1 for c in occ]
    return m, k, dims, tensors, output


def total_cost(m, dims, tensors, output, merges):
    """Return total scalar-mult cost, or None if the schedule is invalid."""
    if len(merges) != m - 1:
        return None
    live = {}
    count = {}
    for tid in range(m):
        s = set(tensors[tid])
        live[tid] = s
        for i in s:
            count[i] = count.get(i, 0) + 1
    nid = m
    total = 0
    for (a, b) in merges:
        if a == b or a not in live or b not in live:
            return None
        SA = live.pop(a)
        SB = live.pop(b)
        U = SA | SB
        cost = 1
        for i in U:
            cost *= dims[i]
        total += cost
        newset = set()
        for i in U:
            others = count[i] - (1 if i in SA else 0) - (1 if i in SB else 0)
            if output[i] or others > 0:
                newset.add(i)
        for i in SA:
            count[i] -= 1
        for i in SB:
            count[i] -= 1
        for i in newset:
            count[i] = count.get(i, 0) + 1
        live[nid] = newset
        nid += 1
    if len(live) != 1:
        return None
    return total


def mincost_greedy_cost(m, dims, tensors, output):
    """Deterministic min-cost greedy contraction cost (the baseline B).

    At each step contract the live pair minimizing (adjacency, contraction
    cost, resulting size, ids).  This mirrors solutions/trivial.py exactly."""
    live = {}
    count = {}
    for tid in range(m):
        s = set(tensors[tid])
        live[tid] = s
        for i in s:
            count[i] = count.get(i, 0) + 1
    nid = m
    total = 0
    while len(live) > 1:
        ids = sorted(live)
        best = None
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                a, b = ids[x], ids[y]
                SA, SB = live[a], live[b]
                U = SA | SB
                cost = 1
                for i in U:
                    cost *= dims[i]
                newset = set()
                for i in U:
                    others = count[i] - (1 if i in SA else 0) - (1 if i in SB else 0)
                    if output[i] or others > 0:
                        newset.add(i)
                rsize = 1
                for i in newset:
                    rsize *= dims[i]
                shared = 0 if (SA & SB) else 1
                key = (shared, cost, rsize, a, b)
                if best is None or key < best[0]:
                    best = (key, a, b, newset, cost)
        _, a, b, newset, cost = best
        SA = live.pop(a)
        SB = live.pop(b)
        for i in SA:
            count[i] -= 1
        for i in SB:
            count[i] -= 1
        for i in newset:
            count[i] = count.get(i, 0) + 1
        live[nid] = newset
        nid += 1
        total += cost
    return total


def fail(reason):
    print("Invalid: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    m, k, dims, tensors, output = read_instance(in_path)

    # --- parse participant output strictly (ints only; reject nan/inf/garbage) ---
    try:
        with open(out_path) as f:
            raw = f.read().split()
    except Exception:
        fail("cannot read output")
    if len(raw) != 2 * (m - 1):
        fail("expected %d integers, got %d" % (2 * (m - 1), len(raw)))
    vals = []
    for tok in raw:
        try:
            v = int(tok)
        except ValueError:
            fail("non-integer token %r" % tok)
        vals.append(v)
    merges = [(vals[2 * i], vals[2 * i + 1]) for i in range(m - 1)]
    for (a, b) in merges:
        if a < 0 or b < 0 or a >= 2 * m or b >= 2 * m:
            fail("id out of range")

    F = total_cost(m, dims, tensors, output, merges)
    if F is None or F <= 0:
        fail("infeasible contraction schedule")

    B = mincost_greedy_cost(m, dims, tensors, output)
    if B is None or B <= 0:
        # should never happen; guard anyway
        print("Ratio: 0.0")
        sys.exit(0)

    sc = min(1000.0, 100.0 * B / F)
    print("m=%d cost=%d baseline=%d" % (m, F, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
