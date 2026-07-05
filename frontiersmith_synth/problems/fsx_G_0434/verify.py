#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the two-compartment PK concentration-decay recovery.

- Reads the test id from <in>, then regenerates the ground-truth disposition law,
  the EARLY train sample (for the baseline) and the hidden LATE clearance-tail
  split entirely from that id (the law lives ONLY here).
- Parses the participant's closed-form C(t) expression through a strict AST
  whitelist (rejects imports / attributes / unknown names / non-finite results).
- Scoring is on LOG-concentration (the natural PK metric), minimisation,
  complexity-penalised, with an irreducible noise floor:

      F = heldout_logMSE * (1 + LAMBDA * complexity)
      B = baseline_logMSE * (1 + LAMBDA * 1)     # baseline = constant train-mean log C
      Ratio = min(1000, 100*B/F) / 1000

  A constant reproduces the baseline (~0.1).  Separating the slow terminal rate
  beta so the law extrapolates into the tail drives held-out error down toward
  the noise floor, but the noise floor + the hidden rates keep it below 1.0.
"""
import sys, math, ast, random

LAMBDA = 0.002
T_LO, T_HI = 0.08, 3.0            # early train window (must match gen.py)
TAIL_LO, TAIL_HI = 4.0, 10.0     # hidden late clearance tail (extrapolation)
N_HELD = 200
EPS = 1e-9
ALLOWED_FUNCS = {"exp": math.exp, "log": math.log, "sin": math.sin,
                 "cos": math.cos, "sqrt": math.sqrt, "tanh": math.tanh,
                 "abs": abs}
ALLOWED_VARS = {"t"}
MAX_EXPR_BYTES = 200000


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---- ground truth (identical to gen.py) ----
def coeffs(t):
    rng = random.Random(90001 + t * 7919)
    A = rng.uniform(6.0, 12.0)
    alpha = rng.uniform(1.3, 2.8)
    B = rng.uniform(1.5, 4.0)
    beta = rng.uniform(0.12, 0.35)
    return A, alpha, B, beta


def fval(tt, cf):
    A, alpha, B, beta = cf
    return A * math.exp(-alpha * tt) + B * math.exp(-beta * tt)


def gen_train(t):
    n = 60 - (t - 1) * 4
    cv = 0.06 + (t - 1) * 0.02
    cf = coeffs(t)
    rng = random.Random(500 + t * 104729)
    lo, hi = math.log(T_LO), math.log(T_HI)
    rows = []
    for _ in range(n):
        tt = math.exp(rng.uniform(lo, hi))
        c = fval(tt, cf) * math.exp(rng.gauss(0.0, cv))
        rows.append((tt, c))
    return rows, cf


def gen_held(t, tm):
    """Late clearance tail with irreducible noise scaled to the baseline RMSE."""
    cf = coeffs(t)
    rng = random.Random(777 + t * 20261)
    pts = []
    for _ in range(N_HELD):
        tt = rng.uniform(TAIL_LO, TAIL_HI)
        pts.append([tt, math.log(fval(tt, cf))])   # clean log-concentration
    bmse = sum((p[1] - tm) ** 2 for p in pts) / len(pts)
    beta_n = 0.44 + (t - 1) * 0.010
    sh = beta_n * math.sqrt(max(bmse, 1e-12))
    nrng = random.Random(999 + t * 15485863)
    for p in pts:
        p[1] = p[1] + nrng.gauss(0.0, sh)
    return pts


# ---- strict expression validation ----
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.USub, ast.UAdd,
)


def validate_ast(tree):
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return "disallowed syntax %s" % type(node).__name__
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS):
                return "disallowed call"
            if node.keywords:
                return "kwargs not allowed"
        if isinstance(node, ast.Name):
            if node.id not in ALLOWED_VARS and node.id not in ALLOWED_FUNCS:
                return "unknown name %s" % node.id
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            return "non-numeric constant"
    return None


def complexity(tree):
    return sum(1 for nd in ast.walk(tree)
               if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant)))


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            header = fh.readline().split()
        t = int(header[1])
    except Exception:
        fail("bad instance header")
    if t < 1 or t > 10000:
        fail("bad test id")

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_EXPR_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_EXPR_BYTES:
        fail("output too large")
    expr = raw.decode("utf-8", "replace").strip()
    if not expr:
        fail("empty expression")
    lines = [ln for ln in expr.splitlines() if ln.strip()]
    if len(lines) != 1:
        fail("expression must be a single line")
    expr = lines[0].strip()

    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        fail("parse error")
    reason = validate_ast(tree)
    if reason:
        fail(reason)
    try:
        code = compile(tree, "<expr>", "eval")
    except Exception:
        fail("compile error")

    # ground truth + splits (baseline = constant train-mean log C)
    train, _ = gen_train(t)
    tm = sum(math.log(max(c, EPS)) for _, c in train) / len(train)
    held = gen_held(t, tm)
    cx = complexity(tree)

    se = 0.0
    for tt, yv in held:
        env = {"t": tt}
        env.update(ALLOWED_FUNCS)
        try:
            p = eval(code, {"__builtins__": {}}, env)
        except Exception:
            fail("evaluation error")
        if not isinstance(p, (int, float)) or isinstance(p, bool):
            fail("non-numeric result")
        p = float(p)
        if p != p or p in (float("inf"), float("-inf")):
            fail("non-finite result")
        lp = math.log(max(p, EPS))          # log-concentration; nonpositive -> heavy penalty
        d = lp - yv
        se += d * d
    F_mse = se / len(held)

    B_mse = sum((tm - yv) ** 2 for _, yv in held) / len(held)
    B = B_mse * (1.0 + LAMBDA * 1)
    F = F_mse * (1.0 + LAMBDA * cx)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("heldout_logMSE=%.6f baseline_logMSE=%.6f complexity=%d  Ratio: %.6f"
          % (F_mse, B_mse, cx, sc / 1000.0))


if __name__ == "__main__":
    main()
