#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the variable-star period-luminosity-color-metallicity
symbolic-regression task.

- Reads the test id from <in> (first line), then regenerates the ground-truth
  relation, the TRAIN sample and the HELD-OUT LONG-PERIOD EXTRAPOLATION split
  entirely from that id (the relation lives ONLY here).
- Parses the participant's closed-form expression <out> over {x1,x2,x3} through
  a strict AST whitelist (rejects imports / attributes / unknown names /
  non-finite constants and results / oversized input).
- Score (minimisation, complexity-penalised held-out MSE):
      F = heldout_MSE * (1 + LAMBDA * complexity)
      B = baseline_MSE * (1 + LAMBDA * 1)      # baseline = constant train mean
      Ratio = min(1000, 100*B/F) / 1000
  A constant reproduces the baseline (~0.1); recovering the P-L curvature +
  color + metallicity structure drives held-out error toward the irreducible
  photometric-noise floor and raises the ratio, but that noise floor plus the
  hidden metallicity-dependent slope keep it well below 1.0.
"""
import sys, math, ast, random

LAMBDA = 0.004
N_HELD = 300
LP_LO, LP_HI = 1.0, 1.8      # held-out long-period extrapolation band (train is [0,1])
COL_LO, COL_HI = 0.55, 1.35
FEH_LO, FEH_HI = -1.5, 0.0
ALLOWED_FUNCS = {"exp": math.exp, "log": math.log, "sin": math.sin,
                 "cos": math.cos, "sqrt": math.sqrt, "tanh": math.tanh,
                 "abs": abs}
ALLOWED_VARS = {"x1", "x2", "x3"}
MAX_EXPR_BYTES = 200000


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---- ground truth (identical to gen.py) ----
def coeffs(t):
    rng = random.Random(310007 + t * 6301)
    return (rng.uniform(-3.4, -2.4), rng.uniform(-0.70, -0.30),
            rng.uniform(0.80, 1.40), rng.uniform(0.15, 0.45),
            rng.uniform(-0.40, -0.10), rng.uniform(-4.60, -3.80))


def fval(x, cf):
    a, q, b, m, d, g = cf
    lp = x[0]
    return a * lp + q * lp * lp + b * x[1] + m * x[2] + d * x[2] * lp + g


def gen_train(t):
    sigma = 0.05 + (t - 1) * 0.03
    n = 220 - (t - 1) * 16
    cf = coeffs(t)
    rng = random.Random(4400 + t * 99991)
    rows = []
    for _ in range(n):
        x = [rng.uniform(0.0, 1.0), rng.uniform(0.5, 1.2), rng.uniform(-1.5, 0.0)]
        rows.append((x, fval(x, cf) + rng.gauss(0.0, sigma)))
    return rows, cf


def gen_held(t, tm):
    """Long-period extrapolation split with an irreducible photometric-noise
    floor scaled to the baseline RMSE (independent of the participant)."""
    cf = coeffs(t)
    rng = random.Random(52049 + t * 20261)
    pts = []
    for _ in range(N_HELD):
        x = [rng.uniform(LP_LO, LP_HI),
             rng.uniform(COL_LO, COL_HI),
             rng.uniform(FEH_LO, FEH_HI)]
        pts.append([x, fval(x, cf)])
    bmse = sum((p[1] - tm) ** 2 for p in pts) / len(pts)
    beta = 0.35 + (t - 1) * 0.015
    sh = beta * math.sqrt(bmse)
    nrng = random.Random(700001 + t * 15485863)
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
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                return "non-numeric constant"
            if not math.isfinite(float(node.value)):
                return "non-finite constant"
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
    for x, yv in held:
        env = {"x1": x[0], "x2": x[1], "x3": x[2]}
        env.update(ALLOWED_FUNCS)
        try:
            p = eval(code, {"__builtins__": {}}, env)
        except Exception:
            fail("evaluation error")
        if not isinstance(p, (int, float)) or isinstance(p, bool):
            fail("non-numeric result")
        p = float(p)
        if not math.isfinite(p):
            fail("non-finite result")
        dd = p - yv
        se += dd * dd
    F_mse = se / len(held)

    B_mse = sum((yv - tm) ** 2 for _, yv in held) / len(held)
    B = B_mse * (1.0 + LAMBDA * 1)
    F = F_mse * (1.0 + LAMBDA * cx)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("heldout_MSE=%.6f baseline_MSE=%.6f complexity=%d  Ratio: %.6f"
          % (F_mse, B_mse, cx, sc / 1000.0))


if __name__ == "__main__":
    main()
