#!/usr/bin/env python3
"""counter.py <in> <out> <ans>  -- deterministic scorer for the shared-addition-sequence
(batch modular exponentiation) problem.  Prints exactly one 'Ratio: <x>' line, exits 0.

Feasibility (strict): first element == 1; every later element is a sum of two earlier
elements; 1 <= a_i <= max(T); every target present; integer tokens only; bounded length.
Objective = minimize L = len(seq)-1.  Baseline B = independent binary chains (no sharing).
score = min(1.0, 0.1 * B / F).
"""
import sys

CAP_TOKENS = 500000       # reject absurdly long outputs
CAP_LEN = 20000           # reject absurdly long *valid-shaped* sequences


def emit(x, reason=""):
    if reason:
        sys.stdout.write("reason: %s\n" % reason)
    sys.stdout.write("Ratio: %.6f\n" % x)
    sys.exit(0)


def bin_cost(t):
    # length of a plain binary (double-and-add) addition chain reaching t
    return (t.bit_length() - 1) + (bin(t).count("1") - 1)


def main():
    if len(sys.argv) < 3:
        emit(0.0, "usage")
    inpath, outpath = sys.argv[1], sys.argv[2]

    # ---- read instance ----
    try:
        idata = open(inpath).read().split()
        k = int(idata[0])
        targets = [int(x) for x in idata[1:1 + k]]
    except Exception:
        emit(0.0, "bad instance")
    if k <= 0 or len(targets) != k:
        emit(0.0, "bad instance count")
    maxt = max(targets)
    tset = set(targets)

    # ---- read participant output ----
    try:
        toks = open(outpath).read().split()
    except Exception:
        emit(0.0, "cannot read output")
    if len(toks) == 0:
        emit(0.0, "empty output")
    if len(toks) > CAP_TOKENS:
        emit(0.0, "output too large")
    seq = []
    for t in toks:
        try:
            v = int(t)          # rejects nan/inf/floats/garbage
        except ValueError:
            emit(0.0, "non-integer token: %s" % t[:20])
        seq.append(v)
    if len(seq) > CAP_LEN:
        emit(0.0, "sequence too long")

    # ---- validate chain ----
    if seq[0] != 1:
        emit(0.0, "first element is not 1")
    present = set()
    present.add(1)
    for i in range(1, len(seq)):
        x = seq[i]
        if x < 1 or x > maxt:
            emit(0.0, "element %d out of range [1,%d]" % (x, maxt))
        ok = False
        for u in present:
            if (x - u) in present:   # u + (x-u) = x, both earlier (x-u may equal u = doubling)
                ok = True
                break
        if not ok:
            emit(0.0, "element %d is not a sum of two earlier elements" % x)
        present.add(x)

    # ---- targets reached ----
    for t in targets:
        if t not in present:
            emit(0.0, "target %d not reached" % t)

    # ---- score ----
    F = len(seq) - 1
    if F <= 0:
        emit(0.0, "degenerate (no additions)")
    B = sum(bin_cost(t) for t in targets)
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    sys.stdout.write("cost F=%d baseline B=%d\n" % (F, B))
    emit(sc / 1000.0)


if __name__ == "__main__":
    main()
