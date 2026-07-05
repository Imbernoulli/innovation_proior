#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the outbreak-dashboard incidence-law recovery task.

- Reads the test id from <in> (first line header), then regenerates the hidden
  incidence law, the TRAIN sample and the HELD-OUT POST-PEAK-TAIL split
  entirely from that id.  The ground-truth law lives ONLY here.
- The held-out split is genuine EXTRAPOLATION: its time axis x1 lies in the
  post-peak tail [11, 17.5], a region the training window [0, 11] never covers.
- Parses the participant's closed-form incidence expression <out> over
  {x1,x2,x3,x4} through a strict AST whitelist (rejects imports/attributes/
  unknown names, non-finite results, oversized input).
- Score (minimisation of complexity-penalised held-out MSE):
      F = heldout_MSE * (1 + LAMBDA * complexity)
      B = baseline_MSE * (1 + LAMBDA * 1)     # baseline = constant train mean
      Ratio = min(1000, 100*B/F) / 1000
  A constant reproduces the baseline (~0.1); recovering the decay law drives
  held-out error down, but an irreducible-noise floor on the tail + the fact
  that the rise/decay rates are only glimpsed pre-tail keep it well below 1.0.
"""
import sys, math, ast, random

LAMBDA = 0.003
TH0, TH1 = 11.0, 17.5    # held-out post-peak-tail window (train is [0, 11])
N_HELD = 300
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
    rng = random.Random(60413 + t * 7919)
    return (rng.uniform(45.0, 85.0), rng.uniform(0.60, 0.90),
            rng.uniform(0.30, 0.48), rng.uniform(6.0, 8.0),
            rng.uniform(-1.5, -0.4), rng.uniform(0.8, 2.2),
            rng.uniform(0.8, 1.6))


def fval(x, cf):
    A, a, b, tp, d, e, g = cf
    u = x[0] - tp
    return A / (math.exp(-a * u) + math.exp(b * u)) + d * x[1] + e * x[2] + g


def gen_train(t):
    sigma = 0.5 + (t - 1) * 0.30
    n = 220 - (t - 1) * 16
    cf = coeffs(t)
    rng = random.Random(400 + t * 104729)
    rows = []
    for _ in range(n):
        x = [rng.uniform(0.0, 11.0), rng.uniform(0.0, 1.0),
             rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0)]
        rows.append((x, fval(x, cf) + rng.gauss(0.0, sigma)))
    return rows, cf


def gen_held(t, tm):
    """Post-peak-tail extrapolation split with irreducible noise scaled to the
    baseline RMSE, so even a perfect law keeps headroom below 1.0."""
    cf = coeffs(t)
    rng = random.Random(701 + t * 20261)
    pts = []
    for _ in range(N_HELD):
        x = [rng.uniform(TH0, TH1), rng.uniform(0.0, 1.0),
             rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0)]
        pts.append([x, fval(x, cf)])
    bmse = sum((p[1] - tm) ** 2 for p in pts) / len(pts)
    beta = 0.40 + (t - 1) * 0.015
    sh = beta * math.sqrt(bmse)
    nrng = random.Random(953 + t * 15485863)
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
