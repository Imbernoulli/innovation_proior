#!/usr/bin/env python3
"""
counter.py <in> <out> <ans> -- Format D checker for the "leaked cipher layer"
XOR straight-line-program (SLP) synthesis problem.

<in>:  m, then m lines of an m-character '0'/'1' string (row i of the target
       GF(2) matrix M).
<out>: K
       K lines "a b"        (new line = line[a] XOR line[b], 1-indexed refs
                              into the m inputs (1..m) plus earlier new lines)
       one line of m indices out_1..out_m (1-indexed line id computing y_i)

Verification: bit-parallel simulate the whole program ONCE using Python big
integers.  Represent every line's value as an m-bit integer VAL where bit
(k-1) is the value that line takes when the input is the k-th standard basis
vector e_k.  Input line x_j therefore starts as VAL = 1 << (j-1) exactly, and
every XOR op is a plain integer XOR.  Because XOR is GF(2)-linear, matching
VAL[out_i] against row i of M (as the SAME k-indexed bitmask) for all m rows
at once certifies y = M x for EVERY x in {0,1}^m simultaneously -- exact
equivalence, no partial credit.

Objective (minimize) = K, the number of XOR ops.
Baseline B = the checker's own per-row independent construction: row i's cost
is max(0, weight(row_i) - 1) (chain the row's own set input bits, no sharing
across rows at all).  ratio = min(1, 0.1 * B / K).
"""
import sys

MAXK_COEF = 60  # generous absolute cap: 60*m*m + 1000 >> any of our tiers' cost


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def main():
    try:
        in_toks = open(sys.argv[1]).read().split()
    except Exception:
        fail("cannot read input")

    it = iter(in_toks)
    try:
        m = int(next(it))
    except Exception:
        fail("bad header")
    if not (2 <= m <= 5000):
        fail("bad m")

    rows_str = []
    try:
        for _ in range(m):
            s = next(it)
            if len(s) != m or any(c not in "01" for c in s):
                fail("bad row token")
            rows_str.append(s)
    except Exception:
        fail("bad matrix body")

    M = []
    for s in rows_str:
        bm = 0
        for j, c in enumerate(s):
            if c == "1":
                bm |= (1 << j)
        M.append(bm)

    if any(bm == 0 for bm in M):
        fail("degenerate zero row in instance")

    # ---- checker's own baseline: independent per-row chain, no sharing ----
    B = 0
    for bm in M:
        w = bin(bm).count("1")
        B += max(0, w - 1)
    if B <= 0:
        fail("degenerate baseline")

    # ---- parse participant output ----
    try:
        out_toks = open(sys.argv[2]).read().split()
    except Exception:
        fail("cannot read output")
    if not out_toks:
        fail("empty output")

    ti = iter(out_toks)
    try:
        K = int(next(ti))
    except Exception:
        fail("bad K")
    if K < 0:
        fail("K < 0")
    if K > MAXK_COEF * m * m + 1000:
        fail("K too large")

    total_lines = m + K  # 1-indexed line ids 1..total_lines
    a = [0] * (K + 1)
    b = [0] * (K + 1)
    try:
        for t in range(1, K + 1):
            av = next(ti)
            bv = next(ti)
            ai = int(av)
            bi = int(bv)
            cur_new_id = m + t
            if not (1 <= ai < cur_new_id) or not (1 <= bi < cur_new_id):
                fail("op %d references out of range / forward line" % t)
            a[t] = ai
            b[t] = bi
    except StopIteration:
        fail("truncated op list")
    except (ValueError, OverflowError):
        fail("non-integer op operand (or non-finite)")

    outrefs = []
    try:
        for _ in range(m):
            ov = next(ti)
            oi = int(ov)
            if not (1 <= oi <= total_lines):
                fail("output ref out of range")
            outrefs.append(oi)
    except StopIteration:
        fail("truncated output-ref list")
    except (ValueError, OverflowError):
        fail("non-integer output ref (or non-finite)")

    # reject trailing garbage tokens (strict schema)
    extra = list(ti)
    if extra:
        fail("trailing tokens after expected schema")

    # ---- bit-parallel simulation (VAL[line] as an m-bit big int) ----
    VAL = [0] * (total_lines + 1)
    for j in range(1, m + 1):
        VAL[j] = 1 << (j - 1)
    for t in range(1, K + 1):
        VAL[m + t] = VAL[a[t]] ^ VAL[b[t]]

    for i in range(m):
        if VAL[outrefs[i]] != M[i]:
            fail("output row %d mismatch (not equivalent to M)" % (i + 1))

    F = K
    ratio = min(1.0, 0.1 * B / max(1e-9, F))
    print("m=%d B=%d K=%d Ratio: %.6f" % (m, B, K, ratio))


if __name__ == "__main__":
    main()
