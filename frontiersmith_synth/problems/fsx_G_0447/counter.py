#!/usr/bin/env python3
"""Deterministic checker for fsx_G_0447 (format D, flop-cse-arith-dag).

CLI: python3 counter.py <in> <out> <ans>   (ans ignored).

The participant submits a STRAIGHT-LINE PROGRAM (arithmetic circuit) that
computes every target polynomial, using only the three scalar binary
operations ADD / SUB / MUL over: input variables x_i, integer constants, and
previously-defined temporaries.  Common-subexpression sharing lets a good
circuit reuse partial products / repeated monomials across the targets.

Output schema (stdout of the solution):
  one instruction per line:   tK OP A B
      tK  = destination temporary, must be t0,t1,... defined in order, once each
      OP  = ADD | SUB | MUL
      A,B = operands: xI (input var), tI (earlier temp), or an integer constant
            (bare integer, optionally #-prefixed)
  then exactly one final line:  OUT o0 o1 ... o_{m-1}
      o_j = the operand whose value must equal target y_j

Scoring (minimization of scalar-op count):
  The checker FIRST verifies EXACT multivariate-polynomial equivalence of each
  produced output to its target (integer arithmetic, no tolerance).  Any
  mismatch / schema violation / non-finite token -> Ratio: 0.0.
  Then F = number of instruction lines (each = one scalar op).
  B = op count of the naive per-term construction the checker builds itself.
  Ratio = min(1, 0.1 * B / F).
"""
import sys

MAX_LINES = 100000       # cap on straight-line program length
MAX_MONO = 200000        # cap on monomials in any intermediate polynomial


def fail(reason):
    print("Invalid: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


# ---------- exact multivariate polynomial arithmetic (dict: exps->int) --------
def p_var(i, n):
    e = [0] * n
    e[i] = 1
    return {tuple(e): 1}


def p_const(c, n):
    if c == 0:
        return {}
    return {tuple([0] * n): int(c)}


def p_add(a, b, sign):
    r = dict(a)
    for k, v in b.items():
        nv = r.get(k, 0) + sign * v
        if nv == 0:
            if k in r:
                del r[k]
        else:
            r[k] = nv
    return r


def p_mul(a, b, n):
    r = {}
    for ka, va in a.items():
        for kb, vb in b.items():
            k = tuple(ka[i] + kb[i] for i in range(n))
            nv = r.get(k, 0) + va * vb
            if nv == 0:
                if k in r:
                    del r[k]
            else:
                r[k] = nv
            if len(r) > MAX_MONO:
                raise OverflowError("polynomial too large")
    return r


# ------------------------------ instance ------------------------------------
def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    n = int(next(it))
    m = int(next(it))
    targets = []
    for _ in range(m):
        K = int(next(it))
        terms = []
        for _ in range(K):
            c = int(next(it))
            d = int(next(it))
            vs = [int(next(it)) for _ in range(d)]
            terms.append((c, vs))
        targets.append(terms)
    return n, m, targets


def target_polys(n, targets):
    polys = []
    for terms in targets:
        acc = {}
        for (c, vs) in terms:
            mono = p_const(c, n)
            for v in vs:
                mono = p_mul(mono, p_var(v, n), n)
            acc = p_add(acc, mono, 1)
        polys.append(acc)
    return polys


def naive_ops(targets):
    """Op count of the naive per-term construction (checker baseline B)."""
    ops = 0
    for terms in targets:
        for (c, vs) in terms:
            d = len(vs)
            ops += max(0, d - 1)          # product chain
            if abs(c) != 1:
                ops += 1                  # coefficient multiply
        ops += max(0, len(terms) - 1)     # accumulation
    return ops


# ------------------------------ participant ---------------------------------
def parse_operand(tok, temps, n):
    if not tok:
        raise ValueError("empty operand")
    c0 = tok[0]
    if c0 == 'x':
        i = int(tok[1:])
        if i < 0 or i >= n:
            raise ValueError("var index out of range")
        return p_var(i, n)
    if c0 == 't':
        i = int(tok[1:])
        if i not in temps:
            raise ValueError("undefined temp")
        return temps[i]
    if c0 == '#':
        return p_const(int(tok[1:]), n)
    # bare integer constant; int() rejects nan/inf/garbage
    return p_const(int(tok), n)


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    n, m, targets = read_instance(in_path)

    try:
        with open(out_path) as f:
            lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
    except Exception:
        fail("cannot read output")

    if not lines:
        fail("empty output")
    # header: a bare integer = number of instruction lines that follow.
    # (int() rejects nan/inf, so a non-finite flood is caught right here.)
    try:
        L = int(lines[0])
    except ValueError:
        fail("first line must be the instruction count")
    if L < 0 or L > MAX_LINES:
        fail("instruction count out of range")
    # exactly: header + L instructions + 1 OUT line
    if len(lines) != L + 2:
        fail("expected %d instruction lines + 1 OUT line, got %d body lines"
             % (L, len(lines) - 1))

    temps = {}          # index -> poly
    next_expected = 0   # temps must be t0, t1, ... in order
    n_instr = 0

    for ln in lines[1:1 + L]:
        parts = ln.split()
        if parts and parts[0] == 'OUT':
            fail("OUT line appeared inside the instruction block")
        # instruction: tK OP A B
        n_instr += 1
        if len(parts) != 4:
            fail("bad instruction arity: %r" % ln)
        dst, op, a, b = parts
        if not (dst.startswith('t') and dst[1:].isdigit()):
            fail("bad destination %r" % dst)
        di = int(dst[1:])
        if di != next_expected:
            fail("temps must be defined in order t0,t1,...; got %r" % dst)
        if op not in ('ADD', 'SUB', 'MUL'):
            fail("unknown op %r" % op)
        try:
            pa = parse_operand(a, temps, n)
            pb = parse_operand(b, temps, n)
        except (ValueError, OverflowError):
            fail("bad operand in %r" % ln)
        except Exception:
            fail("bad operand in %r" % ln)
        try:
            if op == 'ADD':
                res = p_add(pa, pb, 1)
            elif op == 'SUB':
                res = p_add(pa, pb, -1)
            else:
                res = p_mul(pa, pb, n)
        except OverflowError:
            fail("intermediate polynomial too large")
        temps[di] = res
        next_expected += 1

    out_line = lines[1 + L].split()
    if not out_line or out_line[0] != 'OUT':
        fail("missing OUT line")
    outs = out_line[1:]
    if len(outs) != m:
        fail("OUT must list %d operands, got %d" % (m, len(outs)))

    tp = target_polys(n, targets)
    for j, tok in enumerate(outs):
        try:
            pv = parse_operand(tok, temps, n)
        except Exception:
            fail("bad OUT operand %r" % tok)
        if pv != tp[j]:
            fail("target %d not computed exactly" % j)

    F = n_instr
    B = naive_ops(targets)
    if B <= 0:
        B = 1
    if F <= 0:
        # everything computed with zero ops only possible for degenerate
        # single-variable targets; treat as best-possible but bounded.
        F = 1

    sc = min(1000.0, 100.0 * B / F)
    print("n=%d m=%d ops=%d baseline=%d" % (n, m, F, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
