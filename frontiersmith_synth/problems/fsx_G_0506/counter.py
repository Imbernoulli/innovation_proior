#!/usr/bin/env python3
"""Format-D checker for fsx_G_0506.

It parses a candidate arithmetic SLP, verifies exact multivariate polynomial
equivalence for every target, and scores the weighted static operation count.
"""
import re
import sys

ADD_W = 1
MUL_W = 3

MAX_BYTES = 4_000_000
MAX_LINES = 60000
MAX_LINE_CHARS = 512
MAX_TOKEN_CHARS = 40
MAX_CONST = 1_000_000
MAX_ABS_COEF = 10 ** 12
MAX_MONO = 250000
MAX_TOTAL_DEG = 24

INT_RE = re.compile(r"[+-]?\d+\Z")


def fail(reason):
    print("Invalid: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def checked_int(tok, what, lo=None, hi=None):
    if len(tok) > MAX_TOKEN_CHARS or INT_RE.fullmatch(tok) is None:
        fail("bad integer token for %s" % what)
    val = int(tok)
    if lo is not None and val < lo:
        fail("%s below range" % what)
    if hi is not None and val > hi:
        fail("%s above range" % what)
    return val


def checked_coef(c):
    if abs(c) > MAX_ABS_COEF:
        raise OverflowError("coefficient too large")
    return c


def p_const(c, n):
    c = int(c)
    if c == 0:
        return {}
    return {tuple([0] * n): checked_coef(c)}


def p_var(i, n):
    e = [0] * n
    e[i] = 1
    return {tuple(e): 1}


def p_add(a, b, sign=1):
    res = dict(a)
    for key, val in b.items():
        nv = checked_coef(res.get(key, 0) + sign * val)
        if nv:
            res[key] = nv
        elif key in res:
            del res[key]
    if len(res) > MAX_MONO:
        raise OverflowError("polynomial too large")
    return res


def p_mul(a, b, n):
    if not a or not b:
        return {}
    res = {}
    for ea, ca in a.items():
        for eb, cb in b.items():
            exp = tuple(ea[i] + eb[i] for i in range(n))
            if sum(exp) > MAX_TOTAL_DEG:
                raise OverflowError("degree too large")
            nv = checked_coef(res.get(exp, 0) + ca * cb)
            if nv:
                res[exp] = nv
            elif exp in res:
                del res[exp]
            if len(res) > MAX_MONO:
                raise OverflowError("polynomial too large")
    return res


def read_instance(path):
    try:
        toks = open(path, "r", encoding="ascii").read().split()
    except Exception:
        fail("cannot read instance")
    if len(toks) < 2:
        fail("malformed instance")
    idx = 0
    n = checked_int(toks[idx], "n", 1, 64)
    idx += 1
    m = checked_int(toks[idx], "m", 1, 32)
    idx += 1
    targets = []
    for _ in range(m):
        if idx >= len(toks):
            fail("truncated target block")
        k = checked_int(toks[idx], "term count", 1, 1000)
        idx += 1
        terms = []
        for _ in range(k):
            if idx + 2 > len(toks):
                fail("truncated term")
            c = checked_int(toks[idx], "coefficient", -1000, 1000)
            idx += 1
            if c == 0:
                fail("zero coefficient in instance")
            d = checked_int(toks[idx], "degree", 0, 16)
            idx += 1
            if idx + d > len(toks):
                fail("truncated monomial")
            vs = []
            for _ in range(d):
                vs.append(checked_int(toks[idx], "variable index", 0, n - 1))
                idx += 1
            terms.append((c, tuple(vs)))
        targets.append(terms)
    if idx != len(toks):
        fail("trailing instance tokens")
    return n, m, targets


def target_polys(n, targets):
    polys = []
    for terms in targets:
        acc = {}
        for c, vs in terms:
            exp = [0] * n
            for v in vs:
                exp[v] += 1
            mono = {tuple(exp): c}
            acc = p_add(acc, mono, 1)
        polys.append(acc)
    return polys


def baseline_weight(targets):
    total = 0
    for terms in targets:
        for c, vs in terms:
            total += MUL_W * max(0, len(vs) - 1)
            if abs(c) != 1:
                total += MUL_W
        total += ADD_W * max(0, len(terms) - 1)
    return max(1, total)


def parse_operand(tok, temps, n):
    if not tok or len(tok) > MAX_TOKEN_CHARS:
        fail("bad operand token")
    if tok[0] == "x":
        if not tok[1:].isdigit():
            fail("bad variable operand")
        i = int(tok[1:])
        if i < 0 or i >= n:
            fail("variable index out of range")
        return p_var(i, n)
    if tok[0] == "t":
        if not tok[1:].isdigit():
            fail("bad temporary operand")
        i = int(tok[1:])
        if i < 0 or i >= len(temps):
            fail("undefined or future temporary")
        return temps[i]
    raw = tok[1:] if tok.startswith("#") else tok
    c = checked_int(raw, "constant", -MAX_CONST, MAX_CONST)
    return p_const(c, n)


def read_output(path):
    try:
        with open(path, "rb") as f:
            blob = f.read(MAX_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(blob) > MAX_BYTES:
        fail("output too large")
    try:
        text = blob.decode("ascii")
    except UnicodeDecodeError:
        fail("output must be ascii")
    lines = []
    for line in text.splitlines():
        if len(line) > MAX_LINE_CHARS:
            fail("line too long")
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
    if not lines:
        fail("empty output")
    return lines


def main():
    if len(sys.argv) < 3:
        fail("checker expects input and output paths")
    in_path, out_path = sys.argv[1], sys.argv[2]
    n, m, targets = read_instance(in_path)
    want = target_polys(n, targets)

    lines = read_output(out_path)
    header = lines[0].split()
    if len(header) != 1:
        fail("first line must contain only L")
    L = checked_int(header[0], "instruction count", 0, MAX_LINES)
    if len(lines) != L + 2:
        fail("wrong number of instruction lines")

    temps = []
    weighted = 0
    for i in range(L):
        parts = lines[1 + i].split()
        if len(parts) != 4:
            fail("bad instruction arity")
        dst, op, a_tok, b_tok = parts
        if dst != "t%d" % i:
            fail("temporaries must be t0,t1,... in order")
        if op not in ("ADD", "SUB", "MUL"):
            fail("unknown operation")
        pa = parse_operand(a_tok, temps, n)
        pb = parse_operand(b_tok, temps, n)
        try:
            if op == "ADD":
                pc = p_add(pa, pb, 1)
                weighted += ADD_W
            elif op == "SUB":
                pc = p_add(pa, pb, -1)
                weighted += ADD_W
            else:
                pc = p_mul(pa, pb, n)
                weighted += MUL_W
        except OverflowError:
            fail("intermediate polynomial too large")
        temps.append(pc)

    out = lines[-1].split()
    if not out or out[0] != "OUT":
        fail("missing OUT line")
    outs = out[1:]
    if len(outs) != m:
        fail("OUT arity mismatch")
    for j, tok in enumerate(outs):
        got = parse_operand(tok, temps, n)
        if got != want[j]:
            fail("target %d does not match exactly" % j)

    B = baseline_weight(targets)
    F = max(1, weighted)
    ratio = min(1.0, 0.1 * B / F)
    print("weighted_ops=%d baseline=%d instructions=%d" % (weighted, B, L))
    print("Ratio: %.6f" % ratio)


if __name__ == "__main__":
    main()
