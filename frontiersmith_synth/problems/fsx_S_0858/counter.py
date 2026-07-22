import sys

# Format D checker -- straight-line-program (SLP) op-count scorer for a masked
# bit-trick black box.
#
#   1) Parse w and the complete truth table f[0..2^w-1] from <in>.
#   2) Parse the participant's SLP from <out>:
#        K
#        line_1
#        ...
#        line_K
#      Register 0 is the input x (implicit, free). Lines 1..K each define
#      register i by one instruction (see grammar below). All values live in
#      [0, 2^w) -- EVERY instruction's result is taken mod 2^w (fixed-width
#      word semantics, like a real w-bit register), so no separate masking
#      instructions are ever required.
#   3) EXACT-equivalence gate: evaluate register K for every x in [0, 2^w) and
#      require it equal f(x) exactly. Any parse error / bad reference / out-of-
#      range op / mismatch -> Ratio: 0.0.
#   4) Objective (minimize) = number of NON-CONST lines ("ops"; CONST lines are
#      free leaves, like input references). Baseline B = 7 * 2^w (the cost of
#      the naive "one equality-indicator term per table row" construction).
#      Ratio = min(1, 0.1 * B / ops).

MAXLINES = 300
MAXCONST = 1 << 20  # generous bound on raw constant literals before mod-2^w

BINOPS = {"ADD", "SUB", "MUL", "AND", "OR", "XOR"}
SHIFTOPS = {"SHL", "SHR"}
UNOPS = {"NOT"}


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def main():
    inp = open(sys.argv[1]).read().split()
    out = open(sys.argv[2]).read().split()

    it = iter(inp)
    try:
        w = int(next(it))
    except Exception:
        fail("bad header")
    if not (2 <= w <= 6):
        fail("bad w")
    n = 1 << w
    mod = n

    table = []
    try:
        for _ in range(n):
            v = int(next(it))
            if not (0 <= v < mod):
                fail("table value out of range")
            table.append(v)
    except Exception:
        fail("bad table")

    # ---- parse participant SLP ----
    if not out:
        fail("empty output")
    try:
        K = int(out[0])
    except Exception:
        fail("bad K")
    if K < 1 or K > MAXLINES:
        fail("K out of range")

    toks = out[1:]
    pos = 0
    lines = []  # list of (opcode, args...)
    for lineno in range(1, K + 1):
        if pos >= len(toks):
            fail("missing line %d" % lineno)
        opc = toks[pos]; pos += 1
        opc_u = opc.upper()
        if opc_u == "CONST":
            if pos >= len(toks):
                fail("CONST missing operand")
            tk = toks[pos]; pos += 1
            try:
                c = int(tk)
            except Exception:
                fail("CONST non-integer / non-finite operand")
            if abs(c) > MAXCONST:
                fail("CONST out of range")
            lines.append(("CONST", c % mod))
        elif opc_u in BINOPS:
            if pos + 1 >= len(toks):
                fail("%s missing operands" % opc_u)
            try:
                i = int(toks[pos]); j = int(toks[pos + 1])
            except Exception:
                fail("%s non-integer register" % opc_u)
            pos += 2
            if not (0 <= i < lineno and 0 <= j < lineno):
                fail("register out of range / forward reference at line %d" % lineno)
            lines.append((opc_u, i, j))
        elif opc_u in UNOPS:
            if pos >= len(toks):
                fail("%s missing operand" % opc_u)
            try:
                i = int(toks[pos])
            except Exception:
                fail("%s non-integer register" % opc_u)
            pos += 1
            if not (0 <= i < lineno):
                fail("register out of range / forward reference at line %d" % lineno)
            lines.append((opc_u, i))
        elif opc_u in SHIFTOPS:
            if pos + 1 >= len(toks):
                fail("%s missing operands" % opc_u)
            try:
                i = int(toks[pos]); c = int(toks[pos + 1])
            except Exception:
                fail("%s non-integer operand" % opc_u)
            pos += 2
            if not (0 <= i < lineno):
                fail("register out of range / forward reference at line %d" % lineno)
            if not (0 <= c <= w):
                fail("shift amount out of range at line %d" % lineno)
            lines.append((opc_u, i, c))
        else:
            fail("unknown opcode %r at line %d" % (opc, lineno))
    if pos != len(toks):
        fail("trailing tokens (wrong token count)")

    ops = sum(1 for ln in lines if ln[0] != "CONST")
    if ops < 1:
        fail("zero-op program cannot match a non-constant table")

    def evaluate(x):
        regs = [x % mod]
        for ln in lines:
            op = ln[0]
            if op == "CONST":
                regs.append(ln[1])
            elif op == "ADD":
                regs.append((regs[ln[1]] + regs[ln[2]]) % mod)
            elif op == "SUB":
                regs.append((regs[ln[1]] - regs[ln[2]]) % mod)
            elif op == "MUL":
                regs.append((regs[ln[1]] * regs[ln[2]]) % mod)
            elif op == "AND":
                regs.append(regs[ln[1]] & regs[ln[2]])
            elif op == "OR":
                regs.append(regs[ln[1]] | regs[ln[2]])
            elif op == "XOR":
                regs.append(regs[ln[1]] ^ regs[ln[2]])
            elif op == "NOT":
                regs.append((~regs[ln[1]]) % mod)
            elif op == "SHL":
                regs.append((regs[ln[1]] << ln[2]) % mod if ln[2] < w else 0)
                regs[-1] &= mod - 1
            elif op == "SHR":
                regs.append((regs[ln[1]] >> ln[2]) if ln[2] < w else 0)
            else:
                fail("internal: unhandled opcode")
        return regs[-1]

    for x in range(n):
        got = evaluate(x)
        if got != table[x]:
            fail("mismatch at x=%d (got %d want %d)" % (x, got, table[x]))

    B = 7.0 * n
    ratio = min(1.0, 0.1 * B / ops)
    print("K=%d ops=%d B=%.1f Ratio: %.6f" % (K, ops, B, ratio))


if __name__ == "__main__":
    main()
