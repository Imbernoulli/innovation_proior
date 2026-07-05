#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the LLM-pretraining neural-scaling-law task.

- Reads the test id from <in> (first line), then regenerates the ground-truth
  scaling law, the TRAIN sample (small exploratory runs) and the HELD-OUT
  EXTRAPOLATION split (much LARGER compute + data) entirely from that id.  The
  law lives ONLY here.
- Parses the participant's closed-form loss law <out> over {C, D} through a
  strict AST whitelist (rejects imports/attributes/unknown names, non-finite
  results, oversized input).
- Score (minimisation, complexity-penalised held-out MSE):
      F = heldout_MSE * (1 + LAMBDA * complexity)
      B = baseline_MSE * (1 + LAMBDA * 1)      # baseline = constant train mean
      Ratio = min(1000, 100*B/F) / 1000
  A constant reproduces the baseline (~0.1).  Recovering the power-law shape
  with its irreducible floor extrapolates to larger scale and drives held-out
  error down toward the irreducible-noise floor -- but that noise floor plus the
  fact that the exponents are hidden keep even a strong recovery well below 1.0.
"""
import sys, math, ast, random

LAMBDA = 0.003
N_HELD = 300
ALLOWED_FUNCS = {"exp": math.exp, "log": math.log, "sin": math.sin,
                 "cos": math.cos, "sqrt": math.sqrt, "tanh": math.tanh,
                 "abs": abs}
ALLOWED_VARS = {"C", "D"}
MAX_EXPR_BYTES = 200000


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---- ground truth (identical to gen.py) ----
def coeffs(t):
    rng = random.Random(31337 + t * 6997)
    E = rng.uniform(1.55, 1.95)
    A = rng.uniform(6.0, 14.0)
    alpha = rng.uniform(0.22, 0.40)
    B = rng.uniform(4.0, 10.0)
    beta = rng.uniform(0.22, 0.40)
    return E, A, alpha, B, beta


def fval(C, D, cf):
    E, A, alpha, B, beta = cf
    return E + A * C ** (-alpha) + B * D ** (-beta)


def loguniform(rng, lo, hi):
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def gen_train(t):
    sigma = 0.020 + (t - 1) * 0.010
    n = 120 - (t - 1) * 8
    cf = coeffs(t)
    rng = random.Random(8000 + t * 100003)
    rows = []
    for _ in range(n):
        C = loguniform(rng, 1.0, 80.0)
        D = loguniform(rng, 1.0, 50.0)
        rows.append((C, D, fval(C, D, cf) + rng.gauss(0.0, sigma)))
    return rows, cf


def gen_held(t, tm):
    """Extrapolation split at LARGER compute/data, with irreducible noise
    scaled to the held-out signal spread so a perfect functional form still
    leaves a genuine noise floor."""
    cf = coeffs(t)
    rng = random.Random(424242 + t * 20261)
    pts = []
    for _ in range(N_HELD):
        C = loguniform(rng, 200.0, 4000.0)   # >> train's [1, 80]
        D = loguniform(rng, 80.0, 500.0)     # >> train's [1, 50]
        pts.append([C, D, fval(C, D, cf)])
    bmse = sum((p[2] - tm) ** 2 for p in pts) / len(pts)
    frac = 0.45 + (t - 1) * 0.02             # noise floor grows with difficulty
    sh = frac * math.sqrt(max(1e-12, bmse))
    nrng = random.Random(999983 + t * 15485863)
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
    tm = sum(L for _, _, L in train) / len(train)
    held = gen_held(t, tm)

    cx = complexity(tree)

    se = 0.0
    for C, D, yv in held:
        env = {"C": C, "D": D}
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

    B_mse = sum((yv - tm) ** 2 for _, _, yv in held) / len(held)
    B = B_mse * (1.0 + LAMBDA * 1)
    F = F_mse * (1.0 + LAMBDA * cx)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("heldout_MSE=%.6f baseline_MSE=%.6f complexity=%d  Ratio: %.6f"
          % (F_mse, B_mse, cx, sc / 1000.0))


if __name__ == "__main__":
    main()
