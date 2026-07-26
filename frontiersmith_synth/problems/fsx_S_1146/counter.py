#!/usr/bin/env python3
"""counter.py <in> <out> <ans> -- scores a straight-line arithmetic circuit against
the hidden law F(x,y) = (3*x*y*(x+y)) mod (k*x*y + p) for the per-test secret (k,p).

Contract:
  1. Parse the submitted program strictly (bounded, defensive). Any malformed token,
     out-of-range literal, forward/self reference, division/modulo by zero, or
     magnitude blow-up during evaluation -> infeasible -> "Ratio: 0.0".
  2. Re-evaluate the program EXACTLY (Python big integers, no floats anywhere) on
     every published small-domain table point AND on a held-out set of much larger
     (x,y) pairs that is generated deterministically from testId and NEVER shown to
     the solver. Any mismatch anywhere -> "Ratio: 0.0".
  3. Only if step 1-2 pass completely: score = min(1, 0.1 * BASELINE_OPS / L) where
     L is the instruction count of the submitted circuit.
Always exits 0 (score printed on the last line is trusted only when the checker
exits cleanly), never touches wall-clock/GPU/randomness for scoring.
"""
import sys
import random

M = 14
BASELINE_OPS = 16          # reference "obvious direct-definition" circuit needs this many ops
LMAX = 200                 # max instruction count accepted (safety, never helps to exceed it)
LITCAP = 1000               # max |integer literal| accepted in an operand
VALCAP = 10 ** 30           # max |intermediate value| tolerated during evaluation
OPS = {"ADD", "SUB", "MUL", "DIV", "MOD"}

# Must stay byte-for-byte identical to gen.py's PARAMS.
PARAMS = {
    1: (3, 2, 300),
    2: (3, 5, 800),
    3: (3, 7, 3000),
    4: (5, 3, 9000),
    5: (3, 4, 30000),
    6: (3, 9, 90000),
    7: (2, 6, 250000),
    8: (3, 1, 500000),
    9: (7, 8, 1000000),
    10: (3, 3, 1000000),
}


def true_F(x, y, k, p):
    num = 3 * x * y * (x + y)
    den = k * x * y + p
    return num % den


def held_out_points(testId, scale, n=25):
    rng = random.Random(900000 + testId * 97)
    pts = []
    for _ in range(n):
        x = rng.randint(0, scale)
        y = rng.randint(0, scale)
        pts.append((x, y))
    edge = [
        (0, scale), (scale, 0), (scale, scale),
        (1, scale), (scale, 1),
        (scale, max(0, scale - 1)), (max(0, scale - 1), scale),
        (scale // 3, scale), (scale, scale // 7 + 1),
    ]
    return pts + edge


def fail(reason):
    print("Infeasible: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def parse_int_literal(tok):
    if not tok:
        return None
    s = tok
    neg = False
    if s[0] in "+-":
        neg = s[0] == "-"
        s = s[1:]
    if not s or not s.isdigit():
        return None
    v = int(s)
    if neg:
        v = -v
    if abs(v) > LITCAP:
        return None
    return v


def parse_operand(tok, max_reg_index):
    """Returns ('x',None) / ('y',None) / ('lit',v) / ('reg',j) or None on failure."""
    if tok == "x":
        return ("x", None)
    if tok == "y":
        return ("y", None)
    if tok.startswith("R") and len(tok) > 1 and tok[1:].isdigit():
        j = int(tok[1:])
        if 1 <= j <= max_reg_index:
            return ("reg", j)
        return None
    v = parse_int_literal(tok)
    if v is None:
        return None
    return ("lit", v)


def parse_program(text):
    toks_lines = [ln.split() for ln in text.splitlines()]
    toks_lines = [ln for ln in toks_lines if ln]  # drop blank lines
    if not toks_lines:
        return None, "empty output"
    first = toks_lines[0]
    if len(first) != 1:
        return None, "first line must be a single integer L"
    Lv = parse_int_literal(first[0])
    if Lv is None or Lv < 0 or Lv > LMAX:
        return None, "bad or out-of-range instruction count"
    L = Lv
    if len(toks_lines) < 1 + L + 1:
        return None, "not enough lines for declared L and OUT"
    instrs = []
    for i in range(1, L + 1):
        ln = toks_lines[i]
        if len(ln) != 3:
            return None, f"instruction line {i} must have 3 tokens"
        op, atok, btok = ln
        if op not in OPS:
            return None, f"unknown op {op!r} on line {i}"
        a = parse_operand(atok, i - 1)
        b = parse_operand(btok, i - 1)
        if a is None or b is None:
            return None, f"bad operand on instruction {i}"
        instrs.append((op, a, b))
    out_line = toks_lines[L + 1]
    if len(out_line) != 2 or out_line[0] != "OUT":
        return None, "missing OUT line"
    ref = parse_operand(out_line[1], L)
    if ref is None:
        return None, "bad OUT operand"
    return (L, instrs, ref), None


def resolve(operand, x, y, regs):
    kind, v = operand
    if kind == "x":
        return x
    if kind == "y":
        return y
    if kind == "lit":
        return v
    return regs[v]


def eval_program(prog, x, y):
    """Returns (value, ok). ok=False on div/mod-by-zero or magnitude blow-up."""
    L, instrs, ref = prog
    regs = [None] * (L + 1)
    for i, (op, a, b) in enumerate(instrs, start=1):
        av = resolve(a, x, y, regs)
        bv = resolve(b, x, y, regs)
        if op == "ADD":
            r = av + bv
        elif op == "SUB":
            r = av - bv
        elif op == "MUL":
            r = av * bv
        elif op == "DIV":
            if bv == 0:
                return None, False
            r = av // bv
        else:  # MOD
            if bv == 0:
                return None, False
            r = av % bv
        if abs(r) > VALCAP:
            return None, False
        regs[i] = r
    return resolve(ref, x, y, regs), True


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    with open(inf) as fh:
        in_lines = fh.read().split("\n")
    header = in_lines[0].split()
    testId = int(header[0])
    m_declared = int(header[1])
    if m_declared != M:
        fail("input header M mismatch (corrupt test data)")
    if testId not in PARAMS:
        fail("unknown testId (corrupt test data)")
    k, p, scale = PARAMS[testId]

    table = []
    for x in range(M + 1):
        row = [int(t) for t in in_lines[1 + x].split()]
        if len(row) != M + 1:
            fail("corrupt table row in input (test data error)")
        table.append(row)

    try:
        out_text = open(outf).read()
    except Exception:
        fail("cannot read output file")
        return

    # Reject non-finite / garbage tokens up front (nan, inf, floats, symbols).
    for tok in out_text.split():
        if tok in ("nan", "inf", "-inf", "+inf", "Infinity", "-Infinity", "NaN"):
            fail("non-finite token in output")
        low = tok.lower()
        if low in ("nan", "inf", "-inf"):
            fail("non-finite token in output")

    prog, err = parse_program(out_text)
    if prog is None:
        fail(f"parse error: {err}")
    L, instrs, ref = prog

    # 1) exact match on the full published small-domain table
    for x in range(M + 1):
        for y in range(M + 1):
            val, ok = eval_program(prog, x, y)
            if not ok:
                fail(f"division/modulo by zero or overflow at x={x},y={y}")
            if val != table[x][y]:
                fail(f"mismatch on published table at x={x},y={y}: got {val} want {table[x][y]}")

    # 2) exact match on the private held-out extrapolation set (domain-extrapolation gate)
    for (x, y) in held_out_points(testId, scale):
        want = true_F(x, y, k, p)
        val, ok = eval_program(prog, x, y)
        if not ok:
            fail(f"division/modulo by zero or overflow at held-out x={x},y={y}")
        if val != want:
            fail(f"mismatch on held-out extrapolation point x={x},y={y}: got {val} want {want}")

    ops = max(1.0, float(L))  # an L=0 (constant/identity) circuit is never allowed
                                # to be treated as "free": floor its cost at 1 op
    ratio = min(1000.0, 100.0 * BASELINE_OPS / ops) / 1000.0
    print(f"OK: exact match on table + {len(held_out_points(testId, scale))} held-out points; ops={L}")
    print("Ratio: %.6f" % ratio)
    sys.exit(0)


if __name__ == "__main__":
    main()
