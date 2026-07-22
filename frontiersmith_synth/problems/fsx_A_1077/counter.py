import sys

# Format D checker -- minimal-addition straight-line program (SLP) for a fixed
# linear map y = A x.
#
#   1) Parse the target n x n integer matrix A from <in>.
#   2) Parse the participant's SLP from <out>:
#        L
#        L lines:  idx a op b     (op in {+,-}; idx runs n+1..n+L strictly in
#                                  order; a,b are earlier-defined ids in [0,idx-1])
#        1 line:   out_1 ... out_n  (each an id in [0, n+L])
#      id 0 is the reserved constant 0 (always available, free).
#      ids 1..n are the inputs x_1..x_n (always available, free).
#      ids n+1..n+L are the temporaries defined by the L instructions.
#   3) EXACT-equality gate: evaluate the SLP on all n standard basis vectors
#      x = e_j (exact Python integers).  out_i's value on e_j must equal A[i][j]
#      for every i,j -- this proves y = A x exactly (by linearity).  Any parse
#      error, malformed reference, non-finite/overflow value, wrong token count,
#      or mismatch scores 0.
#   4) Objective (minimize) = L, the number of add/sub instructions used.
#      Baseline B = the naive independent-per-row op count (sum of row nnz).
#      Ratio = min(1, 0.1 * B / L).

MAXL = 20000
VALCAP = 10 ** 7


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def main():
    inp = open(sys.argv[1]).read().split()
    outtoks = open(sys.argv[2]).read().split()

    it = iter(inp)
    try:
        n = int(next(it))
    except Exception:
        fail("bad header")
    if not (1 <= n <= 200):
        fail("bad n")
    A = [[0] * n for _ in range(n)]
    try:
        for i in range(n):
            for j in range(n):
                A[i][j] = int(next(it))
    except Exception:
        fail("bad matrix")

    nnz = [sum(1 for v in row if v != 0) for row in A]
    total_nnz = sum(nnz)
    if total_nnz == 0:
        fail("degenerate zero matrix")
    # Tight independent-per-row baseline: input ids are free references, so a
    # row's cheapest NO-SHARING construction leads with a +1-sign term (free)
    # when one exists (cost = nnz-1), and otherwise must pay one gate to
    # negate a leading -1-sign term via "0 - x" (cost = nnz). This is the
    # true minimum a solver with zero cross-row sharing can achieve, so no
    # no-sharing submission can score above the 0.1 baseline.
    row_cost = []
    for row in A:
        k = sum(1 for v in row if v != 0)
        haspos = any(v == 1 for v in row)
        row_cost.append(max(0, k - 1) if haspos else k)
    B = sum(row_cost)

    # ---- parse participant output ----
    if not outtoks:
        fail("empty output")
    ot = iter(outtoks)

    def next_int(msg):
        try:
            return int(next(ot))
        except Exception:
            fail(msg)

    L = next_int("bad L")
    if L < 0 or L > MAXL:
        fail("L out of range")

    instrs = []  # (idx, a, op, b)
    for k in range(L):
        try:
            idx_tok = next(ot)
            idx = int(idx_tok)
        except StopIteration:
            fail("missing instruction line %d" % (k + 1))
        except Exception:
            fail("bad idx at instruction %d" % (k + 1))
        if idx != n + 1 + k:
            fail("instruction ids must be strictly sequential n+1..n+L (got %d expected %d)"
                 % (idx, n + 1 + k))
        try:
            a_tok = next(ot)
            a = int(a_tok)
        except StopIteration:
            fail("missing operand a at instruction %d" % (k + 1))
        except Exception:
            fail("bad operand a at instruction %d" % (k + 1))
        try:
            op = next(ot)
        except StopIteration:
            fail("missing op at instruction %d" % (k + 1))
        if op not in ("+", "-"):
            fail("op must be + or - at instruction %d (got %r)" % (k + 1, op))
        try:
            b_tok = next(ot)
            b = int(b_tok)
        except StopIteration:
            fail("missing operand b at instruction %d" % (k + 1))
        except Exception:
            fail("bad operand b at instruction %d" % (k + 1))
        if not (0 <= a <= idx - 1) or not (0 <= b <= idx - 1):
            fail("operand out of range at instruction %d" % (k + 1))
        instrs.append((idx, a, op, b))

    outs = []
    for i in range(n):
        v = next_int("missing/bad output reference %d" % (i + 1))
        if not (0 <= v <= n + L):
            fail("output reference %d out of range" % (i + 1))
        outs.append(v)

    # strict format: no trailing tokens
    try:
        next(ot)
        fail("trailing tokens after expected output")
    except StopIteration:
        pass

    # ---- exact reconstruction: evaluate SLP on every standard basis vector ----
    for j in range(n):
        val = [0] * (n + L + 1)
        val[0] = 0
        for k in range(1, n + 1):
            val[k] = 1 if k == j + 1 else 0
        for (idx, a, op, b) in instrs:
            va = val[a]
            vb = val[b]
            v = va + vb if op == "+" else va - vb
            if v > VALCAP or v < -VALCAP:
                fail("intermediate value overflow (instruction id %d)" % idx)
            val[idx] = v
        for i in range(n):
            if val[outs[i]] != A[i][j]:
                fail("reconstruction mismatch at row %d for basis column %d" % (i + 1, j + 1))

    ratio = min(1.0, 0.1 * B / max(1, L))
    print("L=%d B=%d Ratio: %.6f" % (L, B, ratio))


if __name__ == "__main__":
    main()
