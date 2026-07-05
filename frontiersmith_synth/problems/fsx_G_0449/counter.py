#!/usr/bin/env python3
"""Deterministic checker for graphics-pipeline matrix-chain composition (format D).

CLI:  python3 counter.py <in> <out> <ans>     (ans ignored)

The participant submits a STRAIGHT-LINE PROGRAM of matrix operations that must
compute the composite transform  T = S_1 . S_2 . ... . S_L  exactly.  Leaves
0..m-1 are the GIVEN matrices (numbered in the order they appear in the stage
blocks).  Each program line appends a new value with the next id.

OUTPUT SCHEMA (participant, on <out>)
    K
    <op> <a> <b>        (K times)
  op in {MUL, ADD, SUB}; a,b reference already-defined ids (< current id).
  MUL: shape(a)=p x q, shape(b)=q x r  -> p x r, cost p*q*r scalar mults.
  ADD/SUB: shapes equal -> same shape, cost 0 scalar mults.
  The value produced by the LAST line must equal T exactly.

SCORING (minimization of scalar multiplies)
  B = multiplies used by the naive construction (materialize each stage densely
      in the given order, then fold left-to-right) -- the checker builds it.
  F = participant multiplies.
  Ratio = min(1, 0.1 * B / F).       trivial (== naive) -> 0.1 ; 10x -> 1.0 .
Any feasibility violation (bad schema, out-of-range id, shape mismatch, wrong
result, non-integer / nan / inf token) -> "Ratio: 0.0".
"""
import sys

MAXK = 20000          # cap on number of program lines
MAXELEM = 1 << 20     # cap on entries of any intermediate matrix


def fail(reason):
    print("Invalid: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


# ---------- exact integer matrix ops ----------
def matmul(A, B):
    n = len(A); k = len(A[0]); m = len(B[0])
    Bt = B  # B is k x m
    out = [[0] * m for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        Oi = out[i]
        for t in range(k):
            a = Ai[t]
            if a == 0:
                continue
            Bt_t = Bt[t]
            for j in range(m):
                Oi[j] += a * Bt_t[j]
    return out


def matadd(A, B, sign):
    n = len(A); m = len(A[0])
    return [[A[i][j] + sign * B[i][j] for j in range(m)] for i in range(n)]


# ---------- instance parser ----------
def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)

    def nxt():
        return next(it)

    L = int(nxt())
    dims = [int(nxt()) for _ in range(L + 1)]
    inputs = []          # flattened given matrices, in appearance order
    stages = []          # (type, din, dout, r, leafdict)

    def take(r, c):
        M = [[int(nxt()) for _ in range(c)] for _ in range(r)]
        idx = len(inputs)
        inputs.append(M)
        return idx

    for i in range(L):
        din, dout = dims[i], dims[i + 1]
        typ = nxt()
        if typ == "DENSE":
            di = take(din, dout)
            stages.append(("DENSE", din, dout, None, {"D": di}))
        elif typ == "LOWRANK":
            r = int(nxt())
            ui = take(din, r)
            vi = take(r, dout)
            stages.append(("LOWRANK", din, dout, r, {"U": ui, "V": vi}))
        elif typ == "SUMLR":
            r = int(nxt())
            ai = take(din, dout)
            bi = take(din, r)
            ci = take(r, dout)
            stages.append(("SUMLR", din, dout, r, {"A": ai, "B": bi, "C": ci}))
        else:
            raise ValueError("bad stage type %r" % typ)
    return L, dims, stages, inputs


def stage_dense(stages, inputs, i):
    """Exact dense matrix of stage i."""
    typ, din, dout, r, leaf = stages[i]
    if typ == "DENSE":
        return inputs[leaf["D"]]
    if typ == "LOWRANK":
        return matmul(inputs[leaf["U"]], inputs[leaf["V"]])
    # SUMLR
    return matadd(inputs[leaf["A"]], matmul(inputs[leaf["B"]], inputs[leaf["C"]]), 1)


def target_matrix(L, stages, inputs):
    T = stage_dense(stages, inputs, 0)
    for i in range(1, L):
        T = matmul(T, stage_dense(stages, inputs, i))
    return T


def baseline_mults(L, dims, stages):
    """Multiplies of the naive construction (== trivial solution)."""
    total = 0
    for i in range(L):
        typ, din, dout, r, _ = stages[i]
        if typ in ("LOWRANK", "SUMLR"):
            total += din * r * dout        # form the low-rank product
    d0 = dims[0]
    for i in range(1, L):                  # fold left-to-right
        total += d0 * dims[i] * dims[i + 1]
    return total


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    L, dims, stages, inputs = read_instance(in_path)
    m = len(inputs)

    T = target_matrix(L, stages, inputs)
    Trows, Tcols = len(T), len(T[0])

    B = baseline_mults(L, dims, stages)
    if B <= 0:
        print("Ratio: 0.0")
        sys.exit(0)

    # ---- parse participant output strictly ----
    try:
        with open(out_path) as f:
            raw = f.read().split()
    except Exception:
        fail("cannot read output")
    if not raw:
        fail("empty output")
    try:
        K = int(raw[0])
    except ValueError:
        fail("K not an integer: %r" % raw[0])
    if K < 1 or K > MAXK:
        fail("K out of range: %d" % K)
    if len(raw) != 1 + 3 * K:
        fail("expected %d tokens, got %d" % (1 + 3 * K, len(raw)))

    # value table: shapes + data, indices 0..m-1 are inputs
    shapes = [(len(M), len(M[0])) for M in inputs]
    data = [M for M in inputs]
    F = 0
    p = 1
    for line in range(K):
        op = raw[p]; sa = raw[p + 1]; sb = raw[p + 2]; p += 3
        if op not in ("MUL", "ADD", "SUB"):
            fail("bad op %r" % op)
        try:
            a = int(sa); b = int(sb)
        except ValueError:
            fail("non-integer operand")
        cur = len(data)
        if not (0 <= a < cur and 0 <= b < cur):
            fail("operand id out of range")
        ra, ca = shapes[a]
        rb, cb = shapes[b]
        if op == "MUL":
            if ca != rb:
                fail("shape mismatch in MUL")
            if ra * cb > MAXELEM:
                fail("intermediate too large")
            data.append(matmul(data[a], data[b]))
            shapes.append((ra, cb))
            F += ra * ca * cb
        else:
            if (ra, ca) != (rb, cb):
                fail("shape mismatch in %s" % op)
            sign = 1 if op == "ADD" else -1
            data.append(matadd(data[a], data[b], sign))
            shapes.append((ra, ca))

    R = data[-1]
    if len(R) != Trows or len(R[0]) != Tcols:
        fail("result shape %dx%d != target %dx%d" % (len(R), len(R[0]), Trows, Tcols))
    for i in range(Trows):
        Ri = R[i]; Ti = T[i]
        for j in range(Tcols):
            if Ri[j] != Ti[j]:
                fail("result != target at (%d,%d)" % (i, j))

    if F <= 0:
        fail("zero multiplies but produced target (impossible)")

    sc = min(1000.0, 100.0 * B / F)
    print("L=%d m=%d mults=%d baseline=%d" % (L, m, F, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
