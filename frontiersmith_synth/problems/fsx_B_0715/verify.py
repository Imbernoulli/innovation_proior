#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the alchemist limiting-reagent symbolic-regression
task.

- Reads the test id from <in> (first line), then regenerates the ground-truth
  recipe, the CALIBRATION sample and the HELD-OUT UNBALANCED-EXTRAPOLATION
  split entirely from that id (the hidden recipe lives ONLY here).
- Parses the participant's closed-form expression <out> over {q1,q2,q3,q4}
  through a strict AST whitelist (rejects imports/attributes/unknown names,
  non-finite results, oversized input).
- Score (minimisation, complexity-penalised held-out MSE):
      F = heldout_MSE * (1 + LAMBDA * complexity)
      B = baseline_MSE * (1 + LAMBDA * 1)      # baseline = constant train mean
      Ratio = min(1000, 100*B/F) / 1000
  A constant reproduces the baseline (~0.1); recovering the true min-law
  shape (including the hidden small-integer stoichiometry) drives held-out
  error toward the irreducible-noise floor and raises the ratio, but the
  noise floor + finite search coarseness keep it below 1.0.
"""
import sys, math, ast, random

K = 4
LAMBDA = 0.003
N_HELD = 200
ALLOWED_FUNCS = {"exp": math.exp, "log": math.log, "sin": math.sin,
                 "cos": math.cos, "sqrt": math.sqrt, "tanh": math.tanh,
                 "abs": abs, "min": min, "max": max}
ALLOWED_VARS = {"q1", "q2", "q3", "q4"}
MAX_EXPR_BYTES = 200000


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---- ground truth (identical logic to gen.py) ----
def coeffs(t):
    rng = random.Random(90101 + t * 7919)
    s_max = 6
    while True:
        s = [rng.randint(1, s_max) for _ in range(K)]
        g0 = s[0]
        for v in s[1:]:
            g0 = math.gcd(g0, v)
        if g0 == 1:
            break
    g = rng.uniform(3.0, 7.0)
    return s, g


def gen_train(t):
    s, g = coeffs(t)
    n = 220 - (t - 1) * 16
    sigma = 0.04 + (t - 1) * 0.025
    rng = random.Random(500 + t * 104729)
    rows = []
    for _ in range(n):
        b = rng.uniform(1.5, 4.5)
        u = [rng.uniform(0.82, 1.18) for _ in range(K)]
        q = [s[i] * b * u[i] for i in range(K)]
        m = min(q[i] / s[i] for i in range(K))
        y_true = g * m
        y = y_true * (1.0 + rng.gauss(0.0, sigma))
        rows.append((q, y))
    return rows, s, g


def gen_held(t, tm):
    """Unbalanced extrapolation split: one reagent kept scarce relative to
    the others, plus an irreducible noise floor scaled to the baseline
    spread (so a constant predictor cannot accidentally score too well)."""
    s, g = coeffs(t)
    rng = random.Random(777 + t * 20261)
    pts = []
    for _ in range(N_HELD):
        idx_lim = rng.randrange(K)
        b_ab = rng.uniform(1.5, 4.5)          # abundant scale, SAME range as training
        frac = rng.uniform(0.10, 0.40)        # limiting reagent's scarcity fraction
        b_lim = b_ab * frac
        u = [rng.uniform(0.9, 1.1) for _ in range(K)]
        q = [0.0] * K
        for i in range(K):
            bi = b_lim if i == idx_lim else b_ab
            q[i] = s[i] * bi * u[i]
        m = min(q[i] / s[i] for i in range(K))
        y_true = g * m
        pts.append([q, y_true])
    bmse = sum((p[1] - tm) ** 2 for p in pts) / len(pts)
    beta = 0.35 + (t - 1) * 0.02
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

    train, s_true, g_true = gen_train(t)
    tm = sum(y for _, y in train) / len(train)
    held = gen_held(t, tm)

    cx = complexity(tree)

    se = 0.0
    for q, yv in held:
        env = {"q1": q[0], "q2": q[1], "q3": q[2], "q4": q[3]}
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
