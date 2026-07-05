import sys

# Format D checker -- minimal-operation linear straight-line program for y = M x.
#   1) Parse target integer matrix M (m x n) from <in>.
#   2) Parse participant's SLP from <out>:
#        P
#        P instructions (DBL a | ADD a b | SUB a b), register n+t created by instr t
#        m output register indices
#   3) Feasibility: operands reference already-defined registers; evaluating each register
#      as an exact integer linear combination of the inputs, output o_i must equal row i of M.
#   4) Objective (minimize) F = P.  Baseline B = binary double-and-add op count.
#      Ratio = min(1.0, 0.1 * B / F).

MAXP = 500000

def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)

# ---- reference binary double-and-add construction (defines baseline B) ----
def baseline_ops(M, m, n):
    class B:
        def __init__(self):
            self.count = 0
        def dbl(self):
            self.count += 1
        def add(self):
            self.count += 1
    b = B()

    def binmul(a):
        # ops to build a*x_j from x_j via the binary method, a >= 1
        bits = bin(a)[2:]
        for bit in bits[1:]:
            b.dbl()
            if bit == '1':
                b.add()

    for i in range(m):
        items = [(j, M[i][j]) for j in range(n) if M[i][j] != 0]
        pos = [(j, c) for (j, c) in items if c > 0]
        neg = [(j, c) for (j, c) in items if c < 0]
        order = pos + neg
        first = True
        for j, c in order:
            binmul(abs(c))
            if first:
                first = False          # running := this term (no combining op)
            else:
                b.add()                # ADD or SUB to combine
    return b.count

def main():
    try:
        inp = open(sys.argv[1]).read().split()
    except Exception:
        fail("no input")
    it = iter(inp)
    try:
        m = int(next(it)); n = int(next(it))
    except Exception:
        fail("bad header")
    if not (1 <= m <= 16 and 1 <= n <= 16):
        fail("bad dims")
    M = [[0] * n for _ in range(m)]
    try:
        for i in range(m):
            for j in range(n):
                M[i][j] = int(next(it))
    except Exception:
        fail("bad matrix")

    B = baseline_ops(M, m, n)
    if B <= 0:
        B = 1

    # ---- parse participant SLP ----
    try:
        out = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")
    if not out:
        fail("empty output")

    def as_int(tok):
        # strict integer; rejects nan/inf/floats
        return int(tok)

    op = iter(out)
    try:
        P = as_int(next(op))
    except Exception:
        fail("bad P")
    if not (0 <= P <= MAXP):
        fail("P out of range")

    # register values as exact integer vectors in Z^n; inputs are unit vectors
    regs = [tuple(1 if k == i else 0 for k in range(n)) for i in range(n)]

    for t in range(P):
        try:
            code = next(op)
        except StopIteration:
            fail("truncated instruction stream")
        cur = len(regs)   # index that this instruction creates
        if code == "DBL":
            try:
                a = as_int(next(op))
            except Exception:
                fail("bad DBL operand")
            if not (0 <= a < cur):
                fail("DBL operand not yet defined")
            regs.append(tuple(2 * v for v in regs[a]))
        elif code == "ADD" or code == "SUB":
            try:
                a = as_int(next(op)); b2 = as_int(next(op))
            except Exception:
                fail("bad ADD/SUB operands")
            if not (0 <= a < cur and 0 <= b2 < cur):
                fail("ADD/SUB operand not yet defined")
            va = regs[a]; vb = regs[b2]
            if code == "ADD":
                regs.append(tuple(x + y for x, y in zip(va, vb)))
            else:
                regs.append(tuple(x - y for x, y in zip(va, vb)))
        else:
            fail("unknown opcode '%s'" % code)

    # read m output register indices
    outs = []
    try:
        for _ in range(m):
            outs.append(as_int(next(op)))
    except Exception:
        fail("missing output register indices")

    total = len(regs)
    for oi in outs:
        if not (0 <= oi < total):
            fail("output register index out of range")

    # exact-equivalence gate
    for i in range(m):
        got = regs[outs[i]]
        want = tuple(M[i])
        if got != want:
            fail("output %d does not equal target row" % i)

    F = P
    if F <= 0:
        # zero instructions can only be correct for unit-vector rows (gen forbids these,
        # but be safe): if it validated above, it is genuinely free -> cap the ratio.
        F = 1
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("ops_yours=%d baseline=%d Ratio: %.6f" % (P, B, sc / 1000.0))

if __name__ == "__main__":
    main()
