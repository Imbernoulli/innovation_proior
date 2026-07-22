import sys
from fractions import Fraction

# Format D checker -- minimum-multiplication straight-line program for a fixed
# even-degree polynomial P(x).
#   1) Parse target coefficients a_0..a_n from <in>.
#   2) Parse participant's straight-line program from <out>:
#         L
#         L lines, each "C p q" / "A i j" / "S i j" / "M i j"
#      register 0 is predefined as x; each line defines register (1..L) in order.
#   3) Symbolically EXPAND the program as an exact rational polynomial in x
#      (never sampled, never floating point) and gate on exact equality to P.
#   4) Objective (minimize) = F = number of M instructions executed.
#      Baseline B = 2n-1 (checker's own naive power-chain-then-scale construction).
#      Ratio = min(1, 0.1 * B / F).

MAXL = 500
MAXTOK = 10 ** 18
MAXDEG_SLACK = 4  # allowed transient degree = MAXDEG_SLACK * n (+ small const)


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def parse_int_tok(tok):
    # int(str) raises ValueError on "nan"/"inf"/"1e3"/garbage -- exactly the
    # non-finite / malformed guard required by the hardening rules.
    if not isinstance(tok, str):
        raise ValueError("bad token")
    return int(tok)


def poly_trim(P):
    while len(P) > 1 and P[-1] == 0:
        P.pop()
    return P


def poly_add(A, B):
    m = max(len(A), len(B))
    A = A + [Fraction(0)] * (m - len(A))
    B = B + [Fraction(0)] * (m - len(B))
    return poly_trim([x + y for x, y in zip(A, B)])


def poly_sub(A, B):
    m = max(len(A), len(B))
    A = A + [Fraction(0)] * (m - len(A))
    B = B + [Fraction(0)] * (m - len(B))
    return poly_trim([x - y for x, y in zip(A, B)])


def poly_mul(A, B):
    C = [Fraction(0)] * (len(A) + len(B) - 1)
    for i, av in enumerate(A):
        if av == 0:
            continue
        for j, bv in enumerate(B):
            if bv:
                C[i + j] += av * bv
    return poly_trim(C)


def main():
    in_toks = open(sys.argv[1]).read().split()
    out_toks = open(sys.argv[2]).read().split()

    it = iter(in_toks)
    try:
        n = int(next(it))
        target = [int(next(it)) for _ in range(n + 1)]
    except Exception:
        fail("bad input (should not happen)")
    if n < 4 or n % 2 != 0 or target[n] == 0:
        fail("bad input (should not happen)")

    if not out_toks:
        fail("empty output")
    try:
        L = parse_int_tok(out_toks[0])
    except Exception:
        fail("bad L")
    if L < 0 or L > MAXL:
        fail("L out of range [0,%d]" % MAXL)
    need = 1 + 3 * L
    if len(out_toks) != need:
        fail("wrong token count (got %d, need %d)" % (len(out_toks), need))

    maxdeg = MAXDEG_SLACK * n + 8

    regs = [[Fraction(0), Fraction(1)]]  # register 0 = x
    mult_count = 0
    pos = 1
    for step in range(L):
        op = out_toks[pos]
        a1, a2 = out_toks[pos + 1], out_toks[pos + 2]
        pos += 3
        if op == "C":
            try:
                p = parse_int_tok(a1)
                q = parse_int_tok(a2)
            except Exception:
                fail("non-finite/unparseable constant at line %d" % (step + 1))
            if q <= 0:
                fail("non-positive denominator at line %d" % (step + 1))
            if abs(p) > MAXTOK or abs(q) > MAXTOK:
                fail("constant magnitude too large at line %d" % (step + 1))
            regs.append([Fraction(p, q)])
        elif op in ("A", "S", "M"):
            try:
                i = parse_int_tok(a1)
                j = parse_int_tok(a2)
            except Exception:
                fail("bad register index at line %d" % (step + 1))
            if not (0 <= i < len(regs)) or not (0 <= j < len(regs)):
                fail("register index out of range at line %d" % (step + 1))
            if op == "A":
                regs.append(poly_add(regs[i], regs[j]))
            elif op == "S":
                regs.append(poly_sub(regs[i], regs[j]))
            else:
                mult_count += 1
                newr = poly_mul(regs[i], regs[j])
                if len(newr) - 1 > maxdeg:
                    fail("degree blowup at line %d" % (step + 1))
                regs.append(newr)
        else:
            fail("unknown opcode %r at line %d" % (op, step + 1))

    result = regs[-1] if L > 0 else regs[0]

    padded = result + [Fraction(0)] * (n + 1 - len(result))
    if len(padded) < n + 1:
        padded += [Fraction(0)] * (n + 1 - len(padded))
    if len(result) - 1 > n:
        fail("result has degree %d > target degree %d" % (len(result) - 1, n))
    for i in range(n + 1):
        if padded[i] != target[i]:
            fail("coefficient mismatch at x^%d" % i)

    F = mult_count
    B = 2 * n - 1
    ratio = min(1.0, 0.1 * B / max(1e-9, F))
    print("F=%d B=%d Ratio: %.6f" % (F, B, ratio))


if __name__ == "__main__":
    main()
