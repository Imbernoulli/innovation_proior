#!/usr/bin/env python3
# counter.py <in> <out> <ans>   (ans ignored) -- deterministic scorer for format D.
#
# Reads a pre-ordered-wire instance and a participant comparator network.
# FIRST verifies exact correctness (the network must sort EVERY input consistent
# with the planted partial order -- checked via the 0-1 principle over all monotone
# 0-1 inputs = all down-sets of the poset), THEN counts the objective:
#     cost F = (#comparators) + alpha * (parallel depth)
# Baseline B = the full insertion-sort network on n wires (ignores the partial order).
# Minimization score:  sc = min(1000, 100 * B / F);  Ratio = sc/1000.
# Any feasibility violation / malformed / non-integer / nan / inf output -> Ratio: 0.0
import sys
from collections import deque

MAX_COMPARATORS = 200000

def fail(msg):
    print("reason: " + msg)
    print("Ratio: 0.0")
    sys.exit(0)

def read_ints_strict(path):
    # Return list of tokens; enforce that every token is a plain integer.
    txt = open(path).read()
    toks = txt.split()
    return toks

def main():
    if len(sys.argv) < 3:
        fail("usage")
    # ---- parse instance ----
    itoks = open(sys.argv[1]).read().split()
    try:
        n = int(itoks[0]); E = int(itoks[1]); alpha = float(itoks[2])
    except Exception:
        fail("bad-instance-header")
    edges = []
    p = 3
    for _ in range(E):
        a = int(itoks[p]); b = int(itoks[p + 1]); p += 2
        edges.append((a, b))

    # ---- enumerate consistent 0-1 inputs = down-sets Z of the poset ----
    preds = [[] for _ in range(n)]
    succ = [[] for _ in range(n)]
    indeg = [0] * n
    for (a, b) in edges:
        preds[b].append(a); succ[a].append(b); indeg[b] += 1
    dq = deque([v for v in range(n) if indeg[v] == 0])
    order = []
    tmp = indeg[:]
    while dq:
        v = dq.popleft(); order.append(v)
        for w in succ[v]:
            tmp[w] -= 1
            if tmp[w] == 0:
                dq.append(w)
    if len(order) != n:
        fail("instance-not-a-dag")  # should never happen for a valid instance

    downsets = []
    def rec(k, Z):
        if k == n:
            downsets.append(Z); return
        v = order[k]
        rec(k + 1, Z)                       # exclude v from Z
        for pr in preds[v]:                 # include v only if all preds in Z
            if not ((Z >> pr) & 1):
                break
        else:
            rec(k + 1, Z | (1 << v))
    rec(0, 0)

    # ---- parse participant network (strict) ----
    otoks = read_ints_strict(sys.argv[2])
    if len(otoks) == 0:
        fail("empty-output")
    vals = []
    for t in otoks:
        # reject anything that is not a plain (optionally signed) integer:
        s = t[1:] if (t and t[0] in "+-") else t
        if not s.isdigit():
            fail("non-integer-token:" + t[:16])
        vals.append(int(t))
    m = vals[0]
    if m < 0 or m > MAX_COMPARATORS:
        fail("bad-comparator-count")
    if len(vals) != 1 + 2 * m:
        fail("token-count-mismatch")
    comps = []
    idx = 1
    for _ in range(m):
        i = vals[idx]; j = vals[idx + 1]; idx += 2
        if not (0 <= i < n and 0 <= j < n):
            fail("index-out-of-range")
        if i >= j:
            fail("comparator-not-ascending")  # require i<j (min->i, max->j)
        comps.append((i, j))

    # ---- feasibility: must sort every consistent 0-1 input ----
    for Z in downsets:
        x = ((1 << n) - 1) & ~Z
        for (i, j) in comps:
            xi = (x >> i) & 1
            xj = (x >> j) & 1
            lo = xi & xj
            hi = xi | xj
            x = (x & ~((1 << i) | (1 << j))) | (lo << i) | (hi << j)
        prev = 0
        for w in range(n):
            b = (x >> w) & 1
            if b < prev:
                fail("does-not-sort")
            prev = b

    # ---- objective ----
    ready = [0] * n
    depth = 0
    for (i, j) in comps:
        d = max(ready[i], ready[j]) + 1
        ready[i] = ready[j] = d
        if d > depth:
            depth = d
    F = m + alpha * depth

    C_base = n * (n - 1) // 2
    D_base = 2 * n - 3 if n >= 2 else 0
    B = C_base + alpha * D_base

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("n=%d comparators=%d depth=%d cost=%.4f baseline=%.4f" % (n, m, depth, F, B))
    print("Ratio: %.6f" % (sc / 1000.0))

if __name__ == "__main__":
    main()
