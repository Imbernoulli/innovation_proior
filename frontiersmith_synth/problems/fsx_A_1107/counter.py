import sys, re, math
from decimal import Decimal, getcontext

# Format-D checker: straight-line-program (arithmetic circuit) op-count scorer
# with a strict extreme-regime accuracy gate.
#
#   1) Parse instance: A B C D E, M, then M grid x-values (<in>).
#   2) Parse participant's straight-line program (<out>):
#        K
#        K lines, each  OP arg1 [arg2]
#      OP in {ADD,SUB,MUL,DIV,ABS,SQRT,EXP,LOG}; args are "x", a finite float
#      literal, or "r<i>" referencing an earlier line's result (1-indexed).
#      The value of the LAST line is the program's output f(x).
#   3) EXACT-equivalence gate (to 1e-9 relative, floored at 1e-300 absolute):
#      evaluate the program (plain double-precision arithmetic, IEEE-style
#      overflow/underflow, no exceptions escape) at EVERY grid x and compare
#      against a high-precision (Decimal, prec=80) ground truth computed from
#      a numerically SAFE closed form (mathematically identical to the naive
#      formula).  ANY mismatch -> Ratio: 0.0.
#   4) Objective (minimize) = weighted op-count of the program, fixed costs:
#        ADD/SUB/MUL/ABS = 1, DIV = 3, SQRT = 4, EXP/LOG = 8.
#      Ratio = min(1, 0.1 * BASELINE_OPS / cost).

getcontext().prec = 80

COST = {"ADD": 1, "SUB": 1, "MUL": 1, "DIV": 3, "ABS": 1, "SQRT": 4, "EXP": 8, "LOG": 8}
UNARY = {"SQRT", "EXP", "LOG", "ABS"}
BINARY = {"ADD", "SUB", "MUL", "DIV"}
MAXK = 400
MAXLIT = 1e18
REL_TOL = 1e-9
ABS_FLOOR = 1e-300
EXP_CLAMP = 750  # beyond this |D*x|, exp(-|D*x|) is unrepresentable in any float64 path -> treat as exactly 0
BASELINE_OPS = 200  # fixed internal baseline op-cost (matches solutions/trivial.py)

REG_RE = re.compile(r"^r([1-9][0-9]*)$")
NONFINITE_WORDS = {"nan", "inf", "+inf", "-inf", "infinity", "-infinity", "+infinity"}


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def ref_f(x, A, B, C, D, E):
    """High-precision ground truth via the mathematically-exact safe closed form."""
    absx = abs(x)
    s = (x * x + B * B).sqrt()
    term1 = (B * B) / (s + absx) + absx - x
    term2 = B / s
    Dx = D * x
    absDx = abs(Dx)
    if absDx > EXP_CLAMP:
        expneg = Decimal(0)
    else:
        expneg = (-absDx).exp()
    pos = Dx if Dx > 0 else Decimal(0)
    term3 = pos + (Decimal(1) + expneg).ln()
    return A * term1 + C * term2 + E * term3


def parse_program(text):
    lines = text.splitlines()
    while lines and lines[-1].strip() == "":
        lines.pop()
    while lines and lines[0].strip() == "":
        lines.pop(0)
    if not lines:
        fail("empty output")
    try:
        K = int(lines[0].strip())
    except Exception:
        fail("bad instruction count header")
    if not (1 <= K <= MAXK):
        fail("K=%d out of range [1,%d]" % (K, MAXK))
    if len(lines) - 1 != K:
        fail("instruction count mismatch: declared %d, found %d body lines" % (K, len(lines) - 1))

    instrs = []
    for idx in range(1, K + 1):
        parts = lines[idx].split()
        if not parts:
            fail("empty instruction line %d" % idx)
        op = parts[0]
        args = parts[1:]
        if op not in COST:
            fail("unknown op '%s' at line %d" % (op, idx))
        need = 1 if op in UNARY else 2
        if len(args) != need:
            fail("arity mismatch for %s at line %d (need %d, got %d)" % (op, idx, need, len(args)))
        parsed = []
        for a in args:
            if a == "x":
                parsed.append(("x", None))
                continue
            m = REG_RE.match(a)
            if m:
                ridx = int(m.group(1))
                if not (1 <= ridx < idx):
                    fail("bad register ref '%s' at line %d (must reference an earlier line)" % (a, idx))
                parsed.append(("r", ridx))
                continue
            if a.lower() in NONFINITE_WORDS:
                fail("non-finite literal '%s' at line %d" % (a, idx))
            try:
                v = float(a)
            except Exception:
                fail("unparsable token '%s' at line %d" % (a, idx))
            if not math.isfinite(v):
                fail("non-finite literal '%s' at line %d" % (a, idx))
            if abs(v) > MAXLIT:
                fail("literal magnitude too large at line %d" % idx)
            parsed.append(("c", v))
        instrs.append((op, parsed))
    cost = sum(COST[op] for op, _ in instrs)
    return instrs, cost


def eval_program(instrs, xf):
    regs = []
    for op, args in instrs:
        vals = []
        for kind, val in args:
            if kind == "x":
                vals.append(xf)
            elif kind == "c":
                vals.append(val)
            else:
                vals.append(regs[val - 1])
        try:
            if op == "ADD":
                r = vals[0] + vals[1]
            elif op == "SUB":
                r = vals[0] - vals[1]
            elif op == "MUL":
                r = vals[0] * vals[1]
            elif op == "DIV":
                try:
                    r = vals[0] / vals[1]
                except ZeroDivisionError:
                    r = float("nan")
            elif op == "ABS":
                r = abs(vals[0])
            elif op == "SQRT":
                try:
                    r = math.sqrt(vals[0])
                except ValueError:
                    r = float("nan")
            elif op == "EXP":
                try:
                    r = math.exp(vals[0])
                except OverflowError:
                    r = float("inf")
            elif op == "LOG":
                try:
                    r = math.log(vals[0])
                except ValueError:
                    r = float("nan")
            else:
                r = float("nan")
        except Exception:
            r = float("nan")
        regs.append(r)
    return regs[-1]


def main():
    try:
        in_toks = open(sys.argv[1]).read().split()
        A_s, B_s, C_s, D_s, E_s = in_toks[0:5]
        M = int(in_toks[5])
        x_toks = in_toks[6:6 + M]
        if len(x_toks) != M:
            fail("malformed instance (input file corrupt)")

        out_text = open(sys.argv[2]).read()
        instrs, cost = parse_program(out_text)

        A_d, B_d, C_d, D_d, E_d = (Decimal(s) for s in (A_s, B_s, C_s, D_s, E_s))

        for xt in x_toks:
            xf = float(xt)
            try:
                computed = eval_program(instrs, xf)
            except Exception:
                fail("evaluator error at x=%r" % xf)
            if not math.isfinite(computed):
                fail("non-finite program output at x=%r" % xf)
            true_f = float(ref_f(Decimal(xt), A_d, B_d, C_d, D_d, E_d))
            thresh = max(REL_TOL * abs(true_f), ABS_FLOOR)
            if abs(computed - true_f) > thresh:
                fail("accuracy gate failed at x=%r: computed=%.17g true=%.17g" % (xf, computed, true_f))

        ratio = min(1.0, 0.1 * BASELINE_OPS / cost)
        print("cost=%d baseline=%d Ratio: %.6f" % (cost, BASELINE_OPS, ratio))
    except SystemExit:
        raise
    except Exception as e:
        print("Ratio: 0.0 (internal error: %r)" % (e,))
    sys.exit(0)


if __name__ == "__main__":
    main()
