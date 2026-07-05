#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  ->  prints 'Ratio: <x in [0,1]>' (last line authoritative).

Deterministic exact scorer for the harbor counterweight manifest problem.
Objective: maximize F = |A-A| / |A+A| (difference-dominant packing).
Baseline B = quality of the first n non-reserved integers (near arithmetic progression, ~1).
Maximization normalization: sc = min(1000, 100 * F / max(1e-9, B)); Ratio = sc / 1000.

Any feasibility violation (wrong count, duplicates, out of range, reserved, non-integer,
non-finite) prints 'Ratio: 0.0' and exits 0.
"""
import sys


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    n = int(toks[0])
    M = int(toks[1])
    k = int(toks[2])
    forb = set(int(x) for x in toks[3:3 + k])
    return n, M, k, forb


def quality(A):
    """Exact |A-A| / |A+A| as a float; A a list of distinct ints."""
    S = set()
    D = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(n):
            aj = A[j]
            D.add(ai - aj)
            if j >= i:
                S.add(ai + aj)
    return len(D) / len(S), len(D), len(S)


def baseline(n, M, forb):
    """First n non-reserved integers 0,1,2,... (near-AP). Returns quality float."""
    A = []
    c = 0
    while len(A) < n:
        if c > M:
            # cannot even build baseline in range; degenerate, avoid div-by-zero
            break
        if c not in forb:
            A.append(c)
        c += 1
    if len(A) < 2:
        return 1.0
    q, _, _ = quality(A)
    return q


def fail(reason):
    sys.stdout.write("reason: %s\nRatio: 0.0\n" % reason)
    sys.exit(0)


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    n, M, k, forb = read_instance(inp)

    # --- parse participant output strictly as integers ---
    try:
        with open(outp) as f:
            raw = f.read().split()
    except Exception:
        fail("cannot read output")

    if len(raw) != n:
        fail("expected exactly %d tokens, got %d" % (n, len(raw)))

    A = []
    for tok in raw:
        # reject floats / nan / inf / anything non-integer
        try:
            v = int(tok)
        except ValueError:
            fail("non-integer token: %r" % tok)
        A.append(v)

    seen = set()
    for v in A:
        if v < 0 or v > M:
            fail("weight %d out of range [0,%d]" % (v, M))
        if v in forb:
            fail("weight %d is reserved" % v)
        if v in seen:
            fail("duplicate weight %d" % v)
        seen.add(v)

    F, dsz, ssz = quality(A)
    B = baseline(n, M, forb)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    ratio = sc / 1000.0
    sys.stdout.write(
        "quality F=%.6f (|A-A|=%d |A+A|=%d) baseline B=%.6f\nRatio: %.6f\n"
        % (F, dsz, ssz, B, ratio)
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
