#!/usr/bin/env python3
"""verify.py <in> <out> <ans>   (ans ignored)

Deterministic scorer for the geothermal-well scaling-law extrapolation problem.

The participant reads the TRAIN rows from <in> and writes ONE closed-form
expression in the variable `x` to <out>. This checker:
  1. re-derives the SAME hidden law (from the field id in <in>) and regenerates a
     HELD-OUT / EXTRAPOLATION split (large-throughput region, disjoint from train)
     deterministically -- the anti-overfit mechanism;
  2. strictly validates the submitted expression (schema, allowed names, finiteness);
  3. scores by held-out RMSE with a mild complexity penalty, normalized against an
     internal flat-persistence baseline B (the "trivial" reference).

Minimization: sc = min(1000, 100 * B / F);  Ratio = sc/1000.
Any feasibility violation -> `Ratio: 0.0`.
"""
import sys, math

# ---- must stay bit-for-bit in sync with gen.py ----
AMP = 0.14
HELD_LO, HELD_HI, N_HELD = 400.0, 3000.0, 24
LAMBDA = 0.02        # complexity penalty per operator token
MAX_EXPR_LEN = 4000

ALLOWED = {
    "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
    "e": math.e, "pi": math.pi,
}


def _u01(a, b, c):
    x = (a * 73856093) ^ (b * 19349663) ^ (c * 83492791)
    x &= 0x7FFFFFFF
    x = (x * 1103515245 + 12345) & 0x7FFFFFFF
    return x / 0x7FFFFFFF


def coeffs(t):
    return 2.0 + 0.3 * t, 50.0 + 10.0 * t, 0.5 + 0.03 * t


def law(x, E, A, al):
    return E + A * x ** (-al)


def held_out(t):
    E, A, al = coeffs(t)
    pts = []
    for i in range(N_HELD):
        lx = math.log(HELD_LO) + (math.log(HELD_HI) - math.log(HELD_LO)) * i / (N_HELD - 1)
        x = math.exp(lx)
        y = law(x, E, A, al) * (1.0 + AMP * (_u01(t, i, 2) - 0.5) * 2.0)
        pts.append((x, y))
    return pts


def bail(reason):
    print("Reason: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    if len(toks) < 2:
        bail("bad instance header")
    t = int(toks[0]); m = int(toks[1])
    ys = []
    idx = 2
    for _ in range(m):
        # x at toks[idx], y at toks[idx+1]
        ys.append(float(toks[idx + 1]))
        idx += 2
    return t, ys


def compile_expr(text):
    # first non-empty, non-comment line only
    expr = None
    for line in text.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            expr = s
            break
    if not expr:
        bail("empty submission")
    if len(expr) > MAX_EXPR_LEN:
        bail("expression too long")
    low = expr.lower()
    for bad in ("__", "import", "lambda", "exec", "eval", ";", "\\"):
        if bad in low:
            bail("disallowed token in expression")
    try:
        code = compile(expr, "<expr>", "eval")
    except Exception as ex:
        bail("parse error: %s" % type(ex).__name__)
    return expr, code


def count_ops(expr):
    n = 0
    for ch in expr:
        if ch in "+-*/":
            n += 1
    for fn in ("exp", "log", "sqrt"):
        n += expr.count(fn)
    return n


def eval_at(code, xval):
    ns = dict(ALLOWED)
    ns["x"] = xval
    v = eval(code, {"__builtins__": {}}, ns)
    return float(v)


def rmse(pred_ys, true_ys):
    s = 0.0
    for p, y in zip(pred_ys, true_ys):
        s += (p - y) ** 2
    return math.sqrt(s / len(true_ys))


def main():
    if len(sys.argv) < 3:
        bail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    t, train_ys = read_instance(inf)
    ho = held_out(t)
    ho_x = [x for x, _ in ho]
    ho_y = [y for _, y in ho]

    # internal baseline B: flat persistence = mean of last 3 train y-values
    if len(train_ys) < 3:
        bail("too few train rows")
    base_const = sum(train_ys[-3:]) / 3.0
    B = rmse([base_const] * len(ho_y), ho_y)
    if not math.isfinite(B) or B <= 0.0:
        bail("degenerate baseline")

    with open(outf) as f:
        text = f.read()
    expr, code = compile_expr(text)

    preds = []
    for x in ho_x:
        try:
            v = eval_at(code, x)
        except Exception as ex:
            bail("evaluation error: %s" % type(ex).__name__)
        if not math.isfinite(v):
            bail("non-finite prediction")
        if abs(v) > 1e12:
            bail("prediction out of range")
        preds.append(v)

    F_raw = rmse(preds, ho_y)
    if not math.isfinite(F_raw):
        bail("non-finite error")
    nops = count_ops(expr)
    F = F_raw * (1.0 + LAMBDA * nops)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("field=%d B=%.6f F_raw=%.6f nops=%d F=%.6f" % (t, B, F_raw, nops, F))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
