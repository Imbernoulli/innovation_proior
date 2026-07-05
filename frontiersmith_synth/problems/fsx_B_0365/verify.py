#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the aquarium sump-plumbing symbolic-regression task.

- Reads the test id from <in> (first line), then regenerates the ground-truth
  return-line flow law, the TRAIN sample and the HELD-OUT FRONTIER
  (extrapolation) split entirely from that id (the law lives ONLY here -- it is
  never printed by gen.py).
- Parses the participant's closed-form expression <out> over {x1,x2,x3,x4}
  through a strict AST whitelist (rejects imports/attributes/unknown names,
  non-finite results, oversized input).
- Score (minimisation, complexity-penalised held-out MSE):
      F = heldout_MSE * (1 + LAMBDA * complexity)
      B = baseline_MSE * (1 + LAMBDA * 1)      # baseline = constant train mean
      Ratio = min(1000, 100*B/F) / 1000
  A constant reproduces the baseline (~0.1); driving held-out error toward the
  irreducible-noise floor raises the ratio, but that floor + the hidden
  exponential head-decay rate keep even a strong recovery well below 1.0.
"""
import sys, math, ast, random

LAMBDA = 0.003
U = 1.4                 # frontier extrapolation upper bound (train is [0,1])
N_HELD = 400
ALLOWED_FUNCS = {"exp": math.exp, "log": math.log, "sin": math.sin,
                 "cos": math.cos, "sqrt": math.sqrt, "tanh": math.tanh,
                 "abs": abs}
ALLOWED_VARS = {"x1", "x2", "x3", "x4"}
MAX_EXPR_BYTES = 200000


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---- ground truth (identical to gen.py) ----
def coeffs(t):
    rng = random.Random(770413 + t * 5087)
    return (rng.uniform(1.5, 3.0), rng.uniform(0.50, 0.90), rng.uniform(1.5, 3.0),
            rng.uniform(1.0, 2.5), rng.uniform(0.5, 1.5), rng.uniform(-0.5, 0.5))


def fval(x, cf):
    a, c, b, d, e, g = cf
    return (a * x[0] * math.exp(c * x[1]) + b * x[2] * x[2]
            + d * x[0] * x[3] + e * x[2] + g)


def gen_train(t):
    sigma = 0.07 + (t - 1) * 0.05
    n = 400 - (t - 1) * 22
    cf = coeffs(t)
    rng = random.Random(1290 + t * 90173)
    rows = []
    for _ in range(n):
        x = [rng.uniform(0.0, 1.0) for _ in range(4)]
        rows.append((x, fval(x, cf) + rng.gauss(0.0, sigma)))
    return rows, cf


def gen_held(t, tm):
    """Frontier split (x_i in [1, U]) with irreducible noise scaled to the
    baseline RMSE so a perfect law still cannot reach Ratio ~ 1.0."""
    cf = coeffs(t)
    rng = random.Random(2027 + t * 33119)
    pts = []
    for _ in range(N_HELD):
        x = [rng.uniform(1.0, U) for _ in range(4)]
        pts.append([x, fval(x, cf)])
    bmse = sum((p[1] - tm) ** 2 for p in pts) / len(pts)
    beta = 0.42 + (t - 1) * 0.02
    sh = beta * math.sqrt(bmse)
    nrng = random.Random(3061 + t * 19486043)
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

    # test id from the instance header
    try:
        with open(inf) as fh:
            header = fh.readline().split()
        t = int(header[1])
    except Exception:
        fail("bad instance header")
    if t < 1 or t > 10000:
        fail("bad test id")

    # participant expression
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

    # ground truth + splits
    train, _ = gen_train(t)
    tm = sum(y for _, y in train) / len(train)
    held = gen_held(t, tm)

    cx = complexity(tree)

    # evaluate participant expression on held-out
    se = 0.0
    for x, yv in held:
        env = {"x1": x[0], "x2": x[1], "x3": x[2], "x4": x[3]}
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
