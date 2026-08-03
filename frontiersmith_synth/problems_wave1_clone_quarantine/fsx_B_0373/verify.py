#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the artisan-bakery supply-chain symbolic-regression
task.

- Reads the test id from <in> (first line), then regenerates the ground-truth
  fresh-yield law, the TRAIN sample and the HELD-OUT FRONTIER (extrapolation)
  split entirely from that id (the law lives ONLY here -- it is never printed by
  gen.py).
- Parses the participant's closed-form expression <out> over {x1,x2,x3,x4}
  through a strict AST whitelist (rejects imports/attributes/unknown names,
  non-finite results, oversized input).
- Score is a COMPLEXITY-PENALISED HELD-OUT R^2, mapped to the harness ratio
  contract.  With the held-out total sum of squares fixed, minimising the
  penalised residual sum of squares is exactly maximising a penalised R^2:
      SS_tot = sum((y - mean_held)^2)                    (fixed by the split)
      SS_res = sum((y - pred)^2)                          (participant)
      R2     = 1 - SS_res / SS_tot
      F = SS_res * (1 + LAMBDA * complexity)              (minimise)
      B = SS_base * (1 + LAMBDA * 1)   # SS of the constant train-mean predictor
      Ratio = min(1000, 100*B/F) / 1000
  A constant reproduces the baseline (~0.1); recovering the hidden envelope
  drives held-out R^2 up, but the irreducible frontier noise + the hidden
  staleness-decay rate keep even a strong recovery well below 1.0.
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
    rng = random.Random(618037 + t * 40961)
    return (rng.uniform(1.5, 3.0), rng.uniform(0.60, 1.10), rng.uniform(1.0, 2.0),
            rng.uniform(1.0, 2.5), rng.uniform(0.5, 1.5), rng.uniform(-0.5, 0.5))


def fval(x, cf):
    a, c, b, d, e, g = cf
    return (a * x[2] * math.exp(-c * x[3]) + b * x[0] * x[1]
            + d * x[1] * x[1] + e * x[0] + g)


def gen_train(t):
    sigma = 0.06 + (t - 1) * 0.045
    n = 380 - (t - 1) * 20
    cf = coeffs(t)
    rng = random.Random(4409 + t * 71993)
    rows = []
    for _ in range(n):
        x = [rng.uniform(0.0, 1.0) for _ in range(4)]
        rows.append((x, fval(x, cf) + rng.gauss(0.0, sigma)))
    return rows, cf


def gen_held(t, tm):
    """Frontier split (x_i in [1, U]) with irreducible noise scaled to the
    baseline RMSE so a perfect law still cannot reach Ratio ~ 1.0."""
    cf = coeffs(t)
    rng = random.Random(5501 + t * 28711)
    pts = []
    for _ in range(N_HELD):
        x = [rng.uniform(1.0, U) for _ in range(4)]
        pts.append([x, fval(x, cf)])
    bmse = sum((p[1] - tm) ** 2 for p in pts) / len(pts)
    beta = 0.40 + (t - 1) * 0.022
    sh = beta * math.sqrt(bmse)
    nrng = random.Random(9187 + t * 15486041)
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
    SS_res = se

    ym = sum(yv for _, yv in held) / len(held)
    SS_tot = sum((yv - ym) ** 2 for _, yv in held)
    SS_base = sum((yv - tm) ** 2 for _, yv in held)

    R2 = 1.0 - SS_res / max(1e-12, SS_tot)

    B = SS_base * (1.0 + LAMBDA * 1)
    F = SS_res * (1.0 + LAMBDA * cx)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("heldout_R2=%.6f SS_res=%.6f SS_base=%.6f complexity=%d  Ratio: %.6f"
          % (R2, SS_res, SS_base, cx, sc / 1000.0))


if __name__ == "__main__":
    main()
