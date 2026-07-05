#!/usr/bin/env python3
"""Deterministic checker for the conflict-free cache-color (3AP-free set) problem.

Usage: python3 verify.py <in> <out> <ans>

<in>  : the instance (a single integer n)
<out> : participant artifact (whitespace-separated integers = the selected set)
<ans> : ignored placeholder

Prints exactly one line ending in "Ratio: <x>" (x in [0,1]). Exits 0.
On ANY feasibility violation prints "Ratio: 0.0" (+ reason) and exits 0.
"""
import sys


def fail(reason):
    print("INVALID (%s) Ratio: 0.0" % reason)
    sys.exit(0)


def base_b_two_digit_count(n, b):
    """Number of integers in [1, n] whose base-b representation uses only digits {0,1}.
    These integers form a 3AP-free set; used as the checker's own reference size."""
    powers = []
    p = 1
    while p <= n:
        powers.append(p)
        p *= b
    L = len(powers)
    cnt = 0
    for mask in range(1, 1 << L):
        s = 0
        m = mask
        i = 0
        while m:
            if m & 1:
                s += powers[i]
            m >>= 1
            i += 1
        if 1 <= s <= n:
            cnt += 1
    return cnt


def has_3ap(values, n):
    """True iff the set of values (all in [1,n]) contains x, x+d, x+2d for some d>=1.
    Bitset method: O(n^2 / 64) worst case with early exit."""
    A = 0
    for v in values:
        A |= 1 << v
    d = 1
    half = n // 2
    while d <= half:
        if A & (A >> d) & (A >> (2 * d)):
            return True
        d += 1
    return False


def main():
    if len(sys.argv) < 3:
        fail("bad args")
    in_path, out_path = sys.argv[1], sys.argv[2]

    # ---- read instance ----
    with open(in_path) as f:
        toks = f.read().split()
    if not toks:
        fail("empty instance")
    try:
        n = int(toks[0])
    except Exception:
        fail("bad instance")
    if not (2 <= n <= 30000):
        fail("n out of range")

    # ---- read participant artifact (bounded, strict) ----
    with open(out_path) as f:
        raw = f.read()
    parts = raw.split()
    # A distinct subset of [1,n] has at most n elements; anything longer is invalid.
    if len(parts) > n:
        fail("too many tokens")
    vals = []
    for tok in parts:
        # strict integer parse: rejects floats, nan, inf, garbage
        try:
            x = int(tok)
        except Exception:
            fail("non-integer token")
        vals.append(x)

    if len(vals) == 0:
        # empty set is feasible but empty -> objective 0
        B = base_b_two_digit_count(n, 5)
        print("empty set. F=0 B=%d Ratio: 0.0" % B)
        sys.exit(0)

    # range + distinctness
    seen = set()
    for x in vals:
        if not (1 <= x <= n):
            fail("value out of range")
        if x in seen:
            fail("duplicate value")
        seen.add(x)

    # no 3-term arithmetic progression
    if has_3ap(vals, n):
        fail("contains 3-term arithmetic progression")

    # ---- score ----
    F = len(vals)
    B = base_b_two_digit_count(n, 5)
    if B <= 0:
        B = 1
    sc = 100.0 * F / max(1e-9, float(B))
    if sc > 1000.0:
        sc = 1000.0
    ratio = sc / 1000.0
    print("valid. F=%d B=%d Ratio: %.6f" % (F, B, ratio))
    sys.exit(0)


if __name__ == "__main__":
    main()
