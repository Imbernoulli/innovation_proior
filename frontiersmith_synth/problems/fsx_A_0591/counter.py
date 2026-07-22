#!/usr/bin/env python3
# counter.py <in> <out> <ans>   (ans is an unused placeholder)
#
# Format D (op-count) checker for "fewest online multiplications to evaluate a fixed
# polynomial". FIRST verifies the submitted straight-line program is a formal identity
# equal to p(x) via EXACT rational (coefficient-wise) arithmetic; THEN counts the
# nonscalar multiplications (MUL lines) and scores against a naive baseline B_hi and a
# baby-step/giant-step reference B_lo. Deterministic; O(input); never times anything.
import sys, math
from fractions import Fraction

LMAX = 20000          # max program lines
TOKMAX = 20000        # max characters in a single token (adversarial guard)


def fail(reason):
    # ANY feasibility violation -> Ratio 0.0
    print("infeasible: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def parse_const(tok):
    # integer or rational p/q; reject nan/inf/garbage.
    if len(tok) > TOKMAX:
        fail("constant too long")
    low = tok.lower()
    if "nan" in low or "inf" in low:
        fail("non-finite constant")
    try:
        if "/" in tok:
            num, den = tok.split("/")
            f = Fraction(int(num), int(den))
        else:
            f = Fraction(int(tok))
    except Exception:
        fail("unparsable constant '%s'" % tok[:32])
    return f


def poly_add(a, b):
    n = max(len(a), len(b))
    r = [Fraction(0)] * n
    for i in range(len(a)):
        r[i] += a[i]
    for i in range(len(b)):
        r[i] += b[i]
    return r


def poly_sub(a, b):
    n = max(len(a), len(b))
    r = [Fraction(0)] * n
    for i in range(len(a)):
        r[i] += a[i]
    for i in range(len(b)):
        r[i] -= b[i]
    return r


def poly_scale(a, c):
    return [x * c for x in a]


def poly_mul(a, b, cap):
    if len(a) - 1 + len(b) - 1 > cap:
        return None  # degree overflow
    r = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        for j, bj in enumerate(b):
            if bj == 0:
                continue
            r[i + j] += ai * bj
    return r


def trim(a):
    # normalize: drop trailing zero coeffs (keep at least [0])
    i = len(a) - 1
    while i > 0 and a[i] == 0:
        i -= 1
    return a[: i + 1]


def ps_ref_count(d):
    # baby-step/giant-step (Paterson-Stockmeyer) nonscalar multiplication count for
    # degree d: precompute x^2..x^k (k-1 mults) + Horner in y=x^k over m blocks (m-1).
    k = 1
    while k * k < d + 1:
        k += 1
    m = (d + 1 + k - 1) // k
    return (k - 1) + (m - 1)


def main():
    if len(sys.argv) < 3:
        fail("usage")
    in_path, out_path = sys.argv[1], sys.argv[2]

    # ---- read instance ----
    with open(in_path) as f:
        toks = f.read().split()
    if not toks:
        fail("empty instance")
    try:
        d = int(toks[0])
        coeffs = [Fraction(int(t)) for t in toks[1 : 1 + d + 1]]
    except Exception:
        fail("bad instance")
    if d < 1 or len(coeffs) != d + 1 or coeffs[d] == 0:
        fail("bad instance shape")
    target = trim(list(coeffs))
    degcap = 2 * d

    # ---- read participant program ----
    try:
        with open(out_path) as f:
            raw = f.readlines()
    except Exception:
        fail("cannot read output")
    if len(raw) > LMAX:
        fail("program too long")

    regs = [[Fraction(0), Fraction(1)]]  # r0 = x  (poly 0 + 1*x)
    mul_count = 0
    out_poly = None
    seen_ret = False

    def getreg(t):
        try:
            idx = int(t)
        except Exception:
            fail("bad register index '%s'" % t[:32])
        if idx < 0 or idx >= len(regs):
            fail("register index out of range")
        return regs[idx]

    for line in raw:
        parts = line.split()
        if not parts:
            continue
        if seen_ret:
            fail("lines after RET")
        op = parts[0].upper()
        if op == "RET":
            if len(parts) != 2:
                fail("RET arity")
            out_poly = getreg(parts[1])
            seen_ret = True
            continue
        if op == "CON":
            if len(parts) != 2:
                fail("CON arity")
            c = parse_const(parts[1])
            regs.append([c])
        elif op == "SMUL":
            if len(parts) != 3:
                fail("SMUL arity")
            a = getreg(parts[1]); c = parse_const(parts[2])
            regs.append(poly_scale(a, c))
        elif op == "SADD":
            if len(parts) != 3:
                fail("SADD arity")
            a = getreg(parts[1]); c = parse_const(parts[2])
            regs.append(poly_add(a, [c]))
        elif op == "ADD":
            if len(parts) != 3:
                fail("ADD arity")
            a = getreg(parts[1]); b = getreg(parts[2])
            regs.append(poly_add(a, b))
        elif op == "SUB":
            if len(parts) != 3:
                fail("SUB arity")
            a = getreg(parts[1]); b = getreg(parts[2])
            regs.append(poly_sub(a, b))
        elif op == "MUL":
            if len(parts) != 3:
                fail("MUL arity")
            a = getreg(parts[1]); b = getreg(parts[2])
            r = poly_mul(a, b, degcap)
            if r is None:
                fail("intermediate degree exceeds 2d")
            regs.append(r)
            mul_count += 1
        else:
            fail("unknown instruction '%s'" % op[:32])
        if len(regs) > LMAX + 4:
            fail("too many registers")

    if not seen_ret or out_poly is None:
        fail("missing RET")

    # ---- exact identity check ----
    got = trim(list(out_poly))
    if got != target:
        fail("program does not compute p(x) identically")

    F = mul_count
    if F <= 0:
        # a polynomial of degree d>=1 needs at least one nonscalar multiply
        fail("zero multiplications cannot yield degree>=1 polynomial")

    # ---- score ----
    B_hi = d * (d - 1) // 2          # naive: recompute every power from scratch
    B_lo = ps_ref_count(d)           # baby-step/giant-step reference
    if B_hi <= B_lo:
        B_hi = B_lo + 1
    lo, hi = math.log(B_lo), math.log(B_hi)
    frac = (hi - math.log(F)) / (hi - lo)
    score = 0.1 + 0.7 * frac
    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0
    print("d=%d F=%d B_hi=%d B_lo=%d" % (d, F, B_hi, B_lo))
    print("Ratio: %.6f" % score)
    sys.exit(0)


if __name__ == "__main__":
    main()
