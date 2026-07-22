#!/usr/bin/env python3
"""
Format-D checker -- shared-subplan query compiler (minimize scalar operations).

Instance (<in>):
    n K
    for each of the K queries:  Tq  then Tq lines "i j coef"
    meaning query Q_q(x) = sum coef * x_i * x_j   (0<=i<=j<n, homogeneous quadratic).

Submission (<out>), a straight-line arithmetic program as a flat token stream:
    P
    then 3*P operand-triples  "OP A B"   (OP in + - * ; defines registers r0..r{P-1})
    then K output operands                (one per query)
Operands:  xI (input i) | rJ (register j, must be defined earlier) | a bare rational
           literal (integer / decimal / p/q ; NOT scientific / nan / inf).

Scoring: FIRST verify each output register equals its target query EXACTLY (polynomial
identity testing over F_p at deterministic secret points).  THEN objective = P (op count,
minimize).  Baseline B = op count of the canonical naive per-query plan (no sharing).
    Ratio = min(1.0, 0.1 * B / P).
"""
import sys
import hashlib
from fractions import Fraction

PRIME = (1 << 61) - 1          # 2**61 - 1, Mersenne prime
NPOINTS = 41                   # PIT evaluation points
PCAP = 20000                   # max instructions
SALT = b"fsx_A_0607::shared-subplan::pit::v1"


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def main():
    raw_in = open(sys.argv[1], "rb").read()
    inp = raw_in.split()
    out = open(sys.argv[2], "rb").read().split()
    out = [tok.decode("ascii", "replace") for tok in out]

    it = iter(inp)

    def nxt_int():
        return int(next(it))

    try:
        n = nxt_int()
        K = nxt_int()
    except Exception:
        fail("bad header")
    if not (1 <= n <= 64 and 1 <= K <= 64):
        fail("bad dims")

    queries = []          # list of list of (i, j, coef)
    try:
        for _ in range(K):
            Tq = nxt_int()
            terms = []
            for _t in range(Tq):
                i = nxt_int()
                j = nxt_int()
                c = nxt_int()
                if not (0 <= i <= j < n):
                    fail("bad monomial index")
                if c == 0:
                    fail("zero coef in instance")
                terms.append((i, j, c))
            queries.append(terms)
    except StopIteration:
        fail("truncated instance")
    except Exception:
        fail("bad instance body")

    # canonical naive baseline op count: per query 2 muls + (Tq-1) adds per term
    B = 0
    for terms in queries:
        Tq = len(terms)
        if Tq >= 1:
            B += 3 * Tq - 1
    if B <= 0:
        fail("degenerate instance (no work)")

    # ---- parse submission ----
    if not out:
        fail("empty output")
    try:
        P = int(out[0])
    except Exception:
        fail("bad P")
    if P < 1:
        fail("P < 1")
    if P > PCAP:
        fail("P too large")
    need = 1 + 3 * P + K
    if len(out) != need:
        fail("wrong token count (got %d, need %d)" % (len(out), need))

    # pre-parse operands into structural form, validating references once.
    # kind 0 = input index, 1 = register index, 2 = literal (mod-p value).
    def parse_operand(tok, max_reg):
        if not tok:
            fail("empty operand")
        h = tok[0]
        if h == 'x':
            try:
                idx = int(tok[1:])
            except Exception:
                fail("bad input operand")
            if not (0 <= idx < n):
                fail("input index out of range")
            return (0, idx)
        if h == 'r':
            try:
                idx = int(tok[1:])
            except Exception:
                fail("bad register operand")
            if not (0 <= idx < max_reg):
                fail("register reference not yet defined")
            return (1, idx)
        # literal
        try:
            fr = Fraction(tok)   # rejects scientific / nan / inf / garbage
        except Exception:
            fail("bad literal operand")
        num = fr.numerator % PRIME
        den = fr.denominator % PRIME
        if den == 0:
            fail("literal denominator divisible by prime")
        val = (num * pow(den, PRIME - 2, PRIME)) % PRIME
        return (2, val)

    instrs = []            # (op_char, opa, opb)
    pos = 1
    for k in range(P):
        op = out[pos]
        a = out[pos + 1]
        b = out[pos + 2]
        pos += 3
        if op not in ('+', '-', '*'):
            fail("bad op")
        oa = parse_operand(a, k)     # may reference r0..r{k-1}
        ob = parse_operand(b, k)
        instrs.append((op, oa, ob))
    outops = [parse_operand(out[pos + q], P) for q in range(K)]

    # ---- polynomial identity testing at deterministic secret points ----
    seed = int.from_bytes(hashlib.sha256(raw_in + SALT).digest(), "big")
    import random
    rng = random.Random(seed)

    def operand_val(kind, xs, regs):
        k, v = kind
        if k == 0:
            return xs[v]
        if k == 1:
            return regs[v]
        return v

    for _pt in range(NPOINTS):
        xs = [rng.randrange(1, PRIME) for _ in range(n)]
        regs = []
        for (op, oa, ob) in instrs:
            va = operand_val(oa, xs, regs)
            vb = operand_val(ob, xs, regs)
            if op == '+':
                regs.append((va + vb) % PRIME)
            elif op == '-':
                regs.append((va - vb) % PRIME)
            else:
                regs.append((va * vb) % PRIME)
        for q in range(K):
            got = operand_val(outops[q], xs, regs)
            tgt = 0
            for (i, j, c) in queries[q]:
                tgt = (tgt + (c % PRIME) * xs[i] % PRIME * xs[j]) % PRIME
            if got % PRIME != tgt % PRIME:
                fail("query %d mismatch" % q)

    ratio = min(1.0, 0.1 * B / max(1, P))
    print("P=%d B=%d Ratio: %.6f" % (P, B, ratio))


if __name__ == "__main__":
    main()
