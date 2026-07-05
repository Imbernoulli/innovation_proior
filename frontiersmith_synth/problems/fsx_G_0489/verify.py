#!/usr/bin/env python3
# verify.py <in> <out> <ans>   deterministic scorer for the admissible-set problem.
# Reads window W from <in>, the participant's set from <out>. Validates feasibility
# strictly (integers, range, distinct, diameter, admissibility); on ANY violation
# prints "Ratio: 0.0" and exits 0. Otherwise scores cardinality against an internal
# double-sieve baseline B. Bit-for-bit deterministic; O(size) fast.
import sys


def primes_upto(n):
    if n < 2:
        return []
    s = bytearray([1]) * (n + 1)
    s[0] = s[1] = 0
    i = 2
    while i * i <= n:
        if s[i]:
            s[i * i::i] = bytearray(len(s[i * i::i]))
        i += 1
    return [i for i in range(2, n + 1) if s[i]]


def double_sieve(W):
    """Coarse reference construction B: from {1..W}, remove residue class 0 mod 2 and
    classes {0,1} mod each odd prime, processing primes in increasing order until the
    prime reaches the surviving count. The result is admissible; its size is the
    normalization baseline."""
    alive = bytearray([1]) * (W + 1)
    alive[0] = 0
    count = W
    for p in primes_upto(W):
        rem = (0,) if p == 2 else (0, 1)
        for r in rem:
            start = r if r != 0 else p
            for m in range(start, W + 1, p):
                if alive[m]:
                    alive[m] = 0
                    count -= 1
        if p >= count:
            break
    return [i for i in range(W + 1) if alive[i]]


def is_admissible(S):
    k = len(S)
    for p in primes_upto(k):
        seen = set()
        for x in S:
            seen.add(x % p)
            if len(seen) == p:
                return False
    return True


def fail(reason):
    print("reason: " + reason)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    inp = open(sys.argv[1]).read().split()
    W = int(inp[0])

    raw = open(sys.argv[2]).read()
    toks = raw.split()
    # bounded read: a feasible answer has at most 3*W+1 distinct values in [0,3W]
    if len(toks) == 0:
        fail("empty output")
    if len(toks) > 4 * W + 100:
        fail("too many tokens")

    vals = []
    for t in toks:
        # strict integer parse; rejects nan/inf/floats/garbage
        try:
            v = int(t)
        except ValueError:
            fail("non-integer token: %r" % t)
        vals.append(v)

    # range + distinctness + diameter
    for v in vals:
        if v < 0 or v > 3 * W:
            fail("value out of range: %d" % v)
    if len(set(vals)) != len(vals):
        fail("duplicate elements")
    S = sorted(set(vals))
    if max(S) - min(S) > W:
        fail("diameter %d exceeds W=%d" % (max(S) - min(S), W))

    if not is_admissible(S):
        fail("set is not admissible")

    F = len(S)
    B = len(double_sieve(W))
    sc = min(1000.0, 100.0 * F / max(1e-9, float(B)))
    print("F=%d B=%d" % (F, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
