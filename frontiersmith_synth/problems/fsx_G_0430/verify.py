#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the antenna rational-transfer-function fit.

- Reads the test id from <in> (first line), then regenerates the hidden
  antenna law, the IN-BAND train sample and the OUT-OF-BAND (roll-off)
  EXTRAPOLATION grading split entirely from that id.  The law lives ONLY here.
- Parses the participant's closed-form expression <out> in the single variable
  f through a strict AST whitelist (rejects imports/attributes/unknown names,
  non-finite results, oversized input).
- Score (minimisation, complexity-penalised held-out MSE):
      F = heldout_MSE * (1 + LAMBDA * complexity)
      B = baseline_MSE * (1 + LAMBDA * 1)      # baseline = constant train mean
      Ratio = min(1000, 100*B/F) / 1000
  A constant reproduces the baseline (~0.1).  Recovering the true rational
  roll-off drives held-out error toward the irreducible-noise floor and raises
  the ratio, but that floor + the hidden (f0, Q) keep it below 1.0.
"""
import sys, math, ast, random

LAMBDA = 0.003
N_HELD = 300
MAX_EXPR_BYTES = 200000
ALLOWED_FUNCS = {"exp": math.exp, "log": math.log, "sin": math.sin,
                 "cos": math.cos, "sqrt": math.sqrt, "tanh": math.tanh,
                 "atan": math.atan, "abs": abs}
ALLOWED_VARS = {"f"}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---- ground truth (identical to gen.py) ----
def coeffs(t):
    rng = random.Random(60413 + t * 7919)
    return (rng.uniform(0.80, 1.30), rng.uniform(2.50, 6.00), rng.uniform(0.80, 2.50))


def presp(f, cf):
    f0, Q, K = cf
    den = (f0 * f0 - f * f) ** 2 + (f0 * f / Q) ** 2
    return K * f * f / den


def gen_train(t):
    cf = coeffs(t)
    f0, Q, K = cf
    peak = K * Q * Q / (f0 * f0)
    sigma = (0.03 + (t - 1) * 0.02) * peak
    n = 120 - (t - 1) * 8
    rng = random.Random(500 + t * 104729)
    rows = []
    for _ in range(n):
        f = rng.uniform(0.15, 2.20)
        rows.append((f, presp(f, cf) + rng.gauss(0.0, sigma)))
    return rows, cf


def gen_held(t, tm):
    """Out-of-band roll-off split (low + high skirts) with irreducible noise
    scaled to the constant-baseline RMSE."""
    cf = coeffs(t)
    rng = random.Random(777 + t * 20261)
    pts = []
    for i in range(N_HELD):
        if i % 2 == 0:
            f = rng.uniform(0.03, 0.14)          # low-frequency skirt
        else:
            f = rng.uniform(2.50, 6.00)          # high-frequency skirt
        pts.append([f, presp(f, cf)])
    bmse = sum((p[1] - tm) ** 2 for p in pts) / len(pts)
    beta = 0.40 + (t - 1) * 0.02
    sh = beta * math.sqrt(max(bmse, 1e-12))
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

    train, _ = gen_train(t)
    tm = sum(y for _, y in train) / len(train)
    held = gen_held(t, tm)
    cx = complexity(tree)

    se = 0.0
    for fx, yv in held:
        env = {"f": fx}
        env.update(ALLOWED_FUNCS)
        try:
            p = eval(code, {"__builtins__": {}}, env)
        except Exception:
            fail("evaluation error")
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            fail("non-numeric result")
        p = float(p)
        if p != p or p in (float("inf"), float("-inf")):
            fail("non-finite result")
        d = p - yv
        se += d * d
    F_mse = se / len(held)

    B_mse = sum((yv - tm) ** 2 for _, yv in held) / len(held)
    B = B_mse * (1.0 + LAMBDA * 1)
    F = F_mse * (1.0 + LAMBDA * cx)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("heldout_MSE=%.6f baseline_MSE=%.6f complexity=%d  Ratio: %.6f"
          % (F_mse, B_mse, cx, sc / 1000.0))


if __name__ == "__main__":
    main()
