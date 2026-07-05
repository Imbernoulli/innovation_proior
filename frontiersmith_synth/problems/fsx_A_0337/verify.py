#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  -- deterministic scorer for sum-frequency coupler wiring.

Reads the instance (n, M) from <in> and the participant's coupler slots from <out>.
Validates feasibility STRICTLY (exactly n distinct integer tokens, each in [0, M],
no floats / nan / inf). Computes F = |A + A| exactly, and an internal baseline
B = |C + C| for the consecutive block C = {0,...,n-1} (= 2n-1). Prints

    ... Ratio: <sc/1000>

where sc = min(1000, 100*F/B). The <ans> placeholder is ignored. Any violation
yields `Ratio: 0.0`. O(n^2), bit-for-bit deterministic.
"""
import sys


def sumset_size(A):
    s = set()
    m = len(A)
    for i in range(m):
        ai = A[i]
        for j in range(i, m):
            s.add(ai + A[j])
    return len(s)


def fail(reason):
    print("reason: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        fail("usage")
    in_path, out_path = sys.argv[1], sys.argv[2]

    # ---- instance ----
    try:
        with open(in_path) as f:
            toks = f.read().split()
        n = int(toks[0]); M = int(toks[1])
    except Exception:
        fail("cannot parse instance")

    # ---- participant output ----
    try:
        with open(out_path) as f:
            raw = f.read().split()
    except Exception:
        fail("cannot read output")

    if len(raw) != n:
        fail("expected exactly n=%d tokens, got %d" % (n, len(raw)))

    A = []
    for tok in raw:
        # strict integer parse: rejects floats, 'nan', 'inf', '1e3', garbage.
        t = tok.strip()
        if not (t.lstrip("+-").isdigit()):
            fail("non-integer token: %r" % tok)
        v = int(t)
        if v < 0 or v > M:
            fail("value %d out of range [0,%d]" % (v, M))
        A.append(v)

    if len(set(A)) != n:
        fail("coupler slots are not distinct")

    # ---- objective ----
    F = sumset_size(A)

    # ---- internal baseline: consecutive block {0,...,n-1} -> |C+C| = 2n-1 ----
    C = list(range(n))
    B = sumset_size(C)          # == 2n - 1
    if B <= 0:
        B = 1

    sc = min(1000.0, 100.0 * F / B)
    print("F=%d B=%d n=%d M=%d" % (F, B, n, M))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
