import sys
from fractions import Fraction

# Format D checker -- minimal-multiplication straight-line evaluation of a fixed
# integer filter polynomial H(x).
#
#   1) Parse target from <in>:  line1 = d ; line2 = d+1 integer taps c[0..d]
#      (c[i] = coeff of x^i, so H(x) = sum_i c[i] x^i).
#   2) Parse participant straight-line program (SLP) from <out>:
#         L
#         L lines, each:  <op> <arg1> <arg2>
#           op   in {mul, add, sub}
#           arg  is  'x'  |  a rational literal 'a' or 'a/b'  |  'r<i>' (i < line idx)
#      The value of the WHOLE program is the result of the LAST instruction.
#   3) EXACT-equivalence gate: the program, as a polynomial in x, must equal H
#      EXACTLY.  Verified by agreement at (2d+1) distinct integer points after a
#      degree-bound guard (a poly of degree <= 2d is fixed by 2d+1 points).
#   4) Objective (MINIMISE) = number of `mul` instructions = F.
#      Baseline B = d  (Horner uses exactly d multiplications).
#      Ratio = min(1, 0.1 * B / F).

MAXL = 512          # max instructions (legit programs are tiny)


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def parse_operand(tok, idx):
    # returns ('var',) | ('const', Fraction) | ('ref', i)
    if tok == 'x':
        return ('var',)
    if tok and tok[0] == 'r' and tok[1:].isdigit():
        i = int(tok[1:])
        if i >= idx:
            return None            # forward / self reference
        return ('ref', i)
    # rational literal; reject anything non-finite (nan/inf won't parse)
    try:
        val = Fraction(tok)
    except (ValueError, ZeroDivisionError):
        return None
    return ('const', val)


def main():
    inp = open(sys.argv[1]).read().split()
    outtxt = open(sys.argv[2]).read().split()

    # ---- parse target polynomial ----
    it = iter(inp)
    try:
        d = int(next(it))
    except StopIteration:
        fail("bad degree")
    if not (1 <= d <= 4096):
        fail("bad degree range")
    coeffs = []
    try:
        for _ in range(d + 1):
            coeffs.append(int(next(it)))
    except (StopIteration, ValueError):
        fail("bad taps")

    # ---- parse participant SLP ----
    if not outtxt:
        fail("empty output")
    try:
        L = int(outtxt[0])
    except ValueError:
        fail("bad L")
    if not (1 <= L <= MAXL):
        fail("L out of range")
    toks = outtxt[1:]
    if len(toks) < 3 * L:
        fail("truncated program")

    ops = []            # (op, operandA, operandB)
    degb = []           # degree upper bound per instruction
    DEGCAP = 2 * d
    for idx in range(L):
        op = toks[3 * idx]
        a = parse_operand(toks[3 * idx + 1], idx)
        b = parse_operand(toks[3 * idx + 2], idx)
        if op not in ('mul', 'add', 'sub') or a is None or b is None:
            fail("bad instruction %d" % idx)

        def opdeg(o):
            if o[0] == 'var':
                return 1
            if o[0] == 'const':
                return 0
            return degb[o[1]]
        da, db = opdeg(a), opdeg(b)
        if op == 'mul':
            dg = da + db
        else:
            dg = max(da, db)
        if dg > DEGCAP:
            fail("degree exceeds cap")
        degb.append(dg)
        ops.append((op, a, b))

    mulcount = sum(1 for op, _, _ in ops if op == 'mul')
    if mulcount == 0:
        fail("no multiplications (cannot realise degree>=1 polynomial)")
    if mulcount > MAXL:
        fail("too many multiplications")

    # ---- exact equivalence via evaluation at 2d+1 distinct integer points ----
    def slp_eval(x0):
        vals = []
        xf = Fraction(x0)
        for (op, a, b) in ops:
            def ev(o):
                if o[0] == 'var':
                    return xf
                if o[0] == 'const':
                    return o[1]
                return vals[o[1]]
            va, vb = ev(a), ev(b)
            if op == 'mul':
                vals.append(va * vb)
            elif op == 'add':
                vals.append(va + vb)
            else:
                vals.append(va - vb)
        return vals[-1]

    def target_eval(x0):
        # Horner over integer taps
        acc = 0
        for c in reversed(coeffs):
            acc = acc * x0 + c
        return acc

    npts = 2 * d + 1
    for x0 in range(0, npts):
        if slp_eval(x0) != Fraction(target_eval(x0)):
            fail("program does not compute H (mismatch at x=%d)" % x0)

    B = d
    F = mulcount
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("muls=%d baseline=%d Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
