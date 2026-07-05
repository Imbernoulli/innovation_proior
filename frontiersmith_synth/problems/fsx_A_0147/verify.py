#!/usr/bin/env python3
"""verify.py <in> <out> <ans>   -- deterministic scorer (ans ignored).

Instance:  "n M".  Participant deploys a set A of integer sensor time-offsets
with |A| <= n and every offset in [0, M].  The "coincidence signature" of an
(ordered-independent) sensor pair {i,j} (including i==j) is offset_i + offset_j.
Objective (MAXIMIZE): the number of DISTINCT signatures, i.e. |A+A| where
A+A = { x + y : x,y in A }.  More distinct signatures  ->  fewer ambiguous
co-triggering events for the monitoring network.

Feasibility (any violation -> Ratio: 0.0):
  * output parses as integers only,
  * 1 <= |A| <= n,
  * all offsets distinct,
  * every offset in [0, M].

Scoring: internal baseline B = |AP+AP| for the arithmetic progression
{0,1,...,n-1} (the densest trivial deployment), whose sumset has size 2n-1.
sc = min(1000, 100 * F / B);  Ratio = sc / 1000  (trivial ~ 0.1).
"""
import sys

def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)

def main():
    inf, outf = sys.argv[1], sys.argv[2]
    with open(inf) as f:
        toks = f.read().split()
    n, M = int(toks[0]), int(toks[1])

    with open(outf) as f:
        raw = f.read().split()
    if not raw:
        fail("empty output")
    vals = []
    for tok in raw:
        try:
            vals.append(int(tok))
        except ValueError:
            fail("non-integer token %r" % tok)

    k = len(vals)
    if k < 1 or k > n:
        fail("|A|=%d not in [1,%d]" % (k, n))
    if len(set(vals)) != k:
        fail("offsets not distinct")
    for x in vals:
        if x < 0 or x > M:
            fail("offset %d out of [0,%d]" % (x, M))

    A = vals
    sums = set()
    for i in range(k):
        ai = A[i]
        for j in range(i, k):
            sums.add(ai + A[j])
    F = len(sums)

    B = 2 * n - 1  # |AP+AP| for {0..n-1}
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("distinct_signatures=%d baseline=%d Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
