#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the wind-tunnel multi-regime drag-law
symbolic-regression task.

- Reads the test id from <in> (first line), then regenerates the hidden
  ground-truth drag law, the MID-Re train sample, and the HIGH-Re EXTRAPOLATION
  held-out split entirely from that id.  The law lives ONLY in this file.
- Parses the participant's closed-form expression <out> over {Re, eps} through a
  strict AST whitelist (rejects imports/attributes/unknown names, non-finite
  results, oversized input).
- Score (minimisation, complexity-penalised held-out MSE):
      F = heldout_MSE * (1 + LAMBDA * complexity)
      B = baseline_MSE * (1 + LAMBDA * 1)      # baseline = constant train mean
      Ratio = min(1000, 100 * B / F) / 1000
  A constant reproduces the baseline (~0.1).  Recovering the viscous /
  boundary-layer / plateau structure drives held-out error down and raises the
  ratio, but the high-Re drag-crisis drop + roughness coupling are only partly
  visible from the mid-Re band, and irreducible balance noise keeps even a
  strong recovery below 1.0.
"""
import sys, math, ast, random

LAMBDA = 0.003
N_HELD = 300
ALLOWED_FUNCS = {"exp": math.exp, "log": math.log, "sin": math.sin,
                 "cos": math.cos, "sqrt": math.sqrt, "tanh": math.tanh,
                 "abs": abs}
ALLOWED_VARS = {"Re", "eps"}
MAX_EXPR_BYTES = 200000


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---- ground truth (identical to gen.py) ----
def coeffs(t):
    rng = random.Random(1300007 + t * 49999)
    a = rng.uniform(18.0, 30.0)
    b = rng.uniform(3.0, 6.0)
    c = rng.uniform(0.35, 0.55)
    h = rng.uniform(0.20, 0.60)
    d = rng.uniform(0.15, 0.35)
    logRc = rng.uniform(3.70, 4.30)
    Rc = 10.0 ** logRc
    m = rng.uniform(2.0, 4.0)
    g = rng.uniform(0.5, 1.5)
    return (a, b, c, h, d, Rc, m, g)


def fval(Re, eps, cf):
    a, b, c, h, d, Rc, m, g = cf
    return (a / Re
            + b / math.sqrt(Re)
            + c + h * eps
            + d / (1.0 + (Re / (Rc * (1.0 + g * eps))) ** m))


def gen_train(t):
    sigma = 0.02 + (t - 1) * 0.012
    n = 220 - (t - 1) * 16
    cf = coeffs(t)
    rng = random.Random(880023 + t * 100003)
    rows = []
    for _ in range(n):
        Re = 10.0 ** rng.uniform(1.5, 3.5)
        eps = rng.uniform(0.0, 0.05)
        Cd = fval(Re, eps, cf) + rng.gauss(0.0, sigma)
        rows.append((Re, eps, Cd))
    return rows, cf


def gen_held(t, tm):
    """HIGH-Re extrapolation split with irreducible noise scaled to baseline."""
    cf = coeffs(t)
    rng = random.Random(660041 + t * 99991)
    pts = []
    for _ in range(N_HELD):
        Re = 10.0 ** rng.uniform(3.60, 5.00)
        eps = rng.uniform(0.0, 0.05)
        pts.append([Re, eps, fval(Re, eps, cf)])
    bmse = sum((p[2] - tm) ** 2 for p in pts) / len(pts)
    beta = 0.28 + (t - 1) * 0.015
    sh = beta * math.sqrt(max(1e-12, bmse))
    nrng = random.Random(550033 + t * 777701)
    for p in pts:
        p[2] = p[2] + nrng.gauss(0.0, sh)
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
    tm = sum(cd for _, _, cd in train) / len(train)
    held = gen_held(t, tm)
    cx = complexity(tree)

    se = 0.0
    for Re, eps, yv in held:
        env = {"Re": Re, "eps": eps}
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
        dd = p - yv
        se += dd * dd
    F_mse = se / len(held)

    B_mse = sum((yv - tm) ** 2 for _, _, yv in held) / len(held)
    B = B_mse * (1.0 + LAMBDA * 1)
    F = F_mse * (1.0 + LAMBDA * cx)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("heldout_MSE=%.6f baseline_MSE=%.6f complexity=%d  Ratio: %.6f"
          % (F_mse, B_mse, cx, sc / 1000.0))


if __name__ == "__main__":
    main()
