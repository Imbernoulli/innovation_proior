#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  -- deterministic checker for fsx_B_1051.

Reads n1,n2 from <in>. Reads a candidate parade (n=n1*n2 lines of "a b") from
<out>. Validates strict feasibility (permutation of Z_n1 x Z_n2), computes
F = P + D (distinct running drifts + distinct consecutive banner-gaps), builds
its own lexicographic-order baseline B the same way, and prints
    Ratio: <F/(10B) capped at 1.0>
On ANY feasibility violation prints Ratio: 0.0 and exits 0.
"""
import sys
import math


def fail(reason):
    print("INFEASIBLE:", reason)
    print("Ratio: 0.0")
    sys.exit(0)


def read_ints_strict(tokens):
    """Parse every token as a base-10 integer; reject anything else
    (non-numeric, floats, nan, inf, huge-but-out-of-range handled later)."""
    out = []
    for tok in tokens:
        # Reject float-looking / nan / inf tokens outright: only accept a
        # (possibly signed) plain integer literal.
        s = tok.strip()
        if s == "" or not (s.lstrip("-").isdigit()):
            return None
        try:
            out.append(int(s))
        except ValueError:
            return None
    return out


def score_of(seq, n1, n2):
    """seq: list of (a,b) tuples, already validated as a full permutation of
    Z_n1 x Z_n2. Returns (P, D, F)."""
    cur = (0, 0)
    sums_seen = set()
    for x in seq:
        cur = ((cur[0] + x[0]) % n1, (cur[1] + x[1]) % n2)
        sums_seen.add(cur)
    P = len(sums_seen)
    diffs_seen = set()
    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i + 1]
        d = ((b[0] - a[0]) % n1, (b[1] - a[1]) % n2)
        diffs_seen.add(d)
    D = len(diffs_seen)
    return P, D, P + D


def main():
    if len(sys.argv) < 3:
        fail("usage: verify.py <in> <out> <ans>")
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path, "r") as f:
        in_tokens = f.read().split()
    if len(in_tokens) < 2:
        fail("bad input file")
    n1, n2 = int(in_tokens[0]), int(in_tokens[1])
    if n1 < 2 or n2 < 2:
        fail("bad input group order")
    n = n1 * n2

    try:
        with open(out_path, "r") as f:
            out_text = f.read()
    except Exception:
        fail("cannot read output file")

    tokens = out_text.split()
    if len(tokens) != 2 * n:
        fail(f"expected {2*n} integer tokens, got {len(tokens)}")

    vals = read_ints_strict(tokens)
    if vals is None:
        fail("non-integer / non-finite token in output")

    for v in vals:
        if not math.isfinite(v):
            fail("non-finite value")

    seq = []
    seen = set()
    for i in range(n):
        a, b = vals[2 * i], vals[2 * i + 1]
        if not (0 <= a < n1 and 0 <= b < n2):
            fail(f"clan ({a},{b}) out of range at position {i+1}")
        pair = (a, b)
        if pair in seen:
            fail(f"duplicate clan ({a},{b}) at position {i+1}")
        seen.add(pair)
        seq.append(pair)

    if len(seen) != n:
        fail("did not cover every clan exactly once")

    P, D, F = score_of(seq, n1, n2)

    # checker's own reference parade: plain lexicographic (house, guild) order
    baseline_seq = [(a, b) for a in range(n1) for b in range(n2)]
    Pb, Db, B = score_of(baseline_seq, n1, n2)
    if B <= 0:
        B = 1e-9

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    ratio = sc / 1000.0
    print(f"n1={n1} n2={n2} n={n} P={P} D={D} F={F} baselineP={Pb} baselineD={Db} B={B}")
    print("Ratio: %.6f" % ratio)
    sys.exit(0)


if __name__ == "__main__":
    main()
