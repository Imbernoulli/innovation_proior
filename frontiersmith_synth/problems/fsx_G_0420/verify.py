#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the factory CES production-function recovery task.

- Reads the test id from <in> (first line), then regenerates the hidden CES law,
  the TRAINING sample, and the HELD-OUT EXTRAPOLATION split entirely from that id
  (the law lives ONLY here -- it is never emitted by gen.py).
- Parses the participant's closed-form expression <out> for output y in the
  variables {K, L} through a strict AST whitelist (rejects imports/attributes/
  unknown names/non-finite results/oversized input).
- Score (minimisation, complexity-penalised held-out MSE):
      F = heldout_MSE * (1 + LAMBDA * complexity)
      B = baseline_MSE * (1 + LAMBDA * 1)      # baseline = constant train-mean y
      Ratio = min(1000, 100*B/F) / 1000
  A constant reproduces the baseline (~0.1).  Recovering the CES shape drives
  held-out error down, but (a) an irreducible productivity-noise floor and
  (b) the fact that any log-polynomial (translog) surrogate is only a Taylor
  approximation of the true CES -- which drifts in the higher-throughput
  extrapolation region -- keep even a strong recovery below 1.0.
"""
import sys, math, ast, random

LAMBDA = 0.003
KLO, KHI = 0.50, 1.50           # training box (must match gen.py)
N_HELD = 300
BETA = 0.42                     # irreducible held-out noise floor (frac of sqrt baseline MSE)
ALLOWED_FUNCS = {"exp": math.exp, "log": math.log, "sin": math.sin,
                 "cos": math.cos, "sqrt": math.sqrt, "tanh": math.tanh,
                 "abs": abs}
ALLOWED_VARS = {"K", "L"}
MAX_EXPR_BYTES = 200000


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---- ground truth (identical to gen.py) ----
def coeffs(t):
    rng = random.Random(60420 + t * 7919)
    A = rng.uniform(0.90, 1.35)
    delta = rng.uniform(0.35, 0.65)
    mag = rng.uniform(0.85, 2.10)
    rho = mag if (t % 2 == 1) else -min(mag, 0.75)
    nu = rng.uniform(0.85, 1.15)
    return A, delta, rho, nu


def fval(K, L, cf):
    A, delta, rho, nu = cf
    base = delta * K ** (-rho) + (1.0 - delta) * L ** (-rho)
    return A * base ** (-nu / rho)


def gen_train_mean(t):
    sigma = 0.03 + (t - 1) * 0.018
    n = 220 - (t - 1) * 15
    cf = coeffs(t)
    rng = random.Random(500 + t * 104729)
    s = 0.0
    for _ in range(n):
        K = rng.uniform(KLO, KHI)
        L = rng.uniform(KLO, KHI)
        y = fval(K, L, cf) * math.exp(rng.gauss(0.0, sigma))
        s += y
    return s / n


def gen_held(t, tm):
    """Held-out EXTRAPOLATION split: input RATIOS pushed outside the training
    range (capital-intensive and labour-intensive expansion scenarios), where
    the elasticity of substitution -- not just returns to scale -- drives output.
    Irreducible productivity noise is scaled to sqrt(baseline MSE) so a strong
    recovery cannot drive held-out error to zero (headroom)."""
    cf = coeffs(t)
    rng = random.Random(777 + t * 20261)
    pts = []
    for i in range(N_HELD):
        if i % 2 == 0:
            # capital-intensive: K high, L low  -> large K/L ratio
            K = rng.uniform(1.80, 2.80)
            L = rng.uniform(0.30, 0.55)
        else:
            # labour-intensive: L high, K low   -> small K/L ratio
            K = rng.uniform(0.30, 0.55)
            L = rng.uniform(1.80, 2.80)
        pts.append([K, L, fval(K, L, cf)])
    bmse = sum((p[2] - tm) ** 2 for p in pts) / len(pts)
    sh = BETA * math.sqrt(bmse)
    nrng = random.Random(999 + t * 15485863)
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

    tm = gen_train_mean(t)
    held = gen_held(t, tm)
    cx = complexity(tree)

    se = 0.0
    for K, L, yv in held:
        env = {"K": K, "L": L}
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

    B_mse = sum((yv - tm) ** 2 for _, _, yv in held) / len(held)
    B = B_mse * (1.0 + LAMBDA * 1)
    F = F_mse * (1.0 + LAMBDA * cx)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("heldout_MSE=%.6f baseline_MSE=%.6f complexity=%d  Ratio: %.6f"
          % (F_mse, B_mse, cx, sc / 1000.0))


if __name__ == "__main__":
    main()
