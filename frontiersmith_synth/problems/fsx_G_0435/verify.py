#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the options-desk implied-vol-surface symbolic
regression task.

- Reads the test id from <in> (first line), then regenerates the ground-truth
  surface, the near-ATM TRAIN sample and the FAR-STRIKE HELD-OUT split entirely
  from that id (the hidden law lives ONLY here).
- Parses the participant's closed-form expression <out> for sigma over {k, t}
  through a strict AST whitelist (rejects imports/attributes/unknown names,
  non-finite results, oversized input).
- Score (minimisation, complexity-penalised held-out MSE):
      F = heldout_MSE * (1 + LAMBDA * complexity)
      B = baseline_MSE * (1 + LAMBDA * 1)      # baseline = constant train mean
      Ratio = min(1000, 100*B/F) / 1000
  A flat-vol constant reproduces the baseline (~0.1); recovering the wing
  convexity drives held-out error down and raises the ratio, but an irreducible
  quote-noise floor + the hidden smile-decay rate keep it below 1.0.
"""
import sys, math, ast, random

LAMBDA = 0.003
KW = 0.45               # far-strike wing bound (train is |k| <= 0.15)
N_HELD = 320
ALLOWED_FUNCS = {"exp": math.exp, "log": math.log, "sin": math.sin,
                 "cos": math.cos, "sqrt": math.sqrt, "tanh": math.tanh,
                 "abs": abs}
ALLOWED_VARS = {"k", "t"}
MAX_EXPR_BYTES = 200000


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---- ground truth (identical to gen.py) ----
def coeffs(t):
    rng = random.Random(60413 + t * 7919)
    return (rng.uniform(0.15, 0.26), rng.uniform(-0.03, 0.05),
            rng.uniform(-0.55, -0.20), rng.uniform(0.60, 1.20),
            rng.uniform(0.50, 1.50), rng.uniform(0.80, 1.80))


def fval(k, t, cf):
    a, term, skew, smile, conv, rate = cf
    return (a + term * t + skew * k + smile * k * k
            + conv * k * k * math.exp(-rate * t))


def gen_train(tid):
    sigma_noise = 0.0040 + (tid - 1) * 0.0016
    n = 200 - (tid - 1) * 14
    cf = coeffs(tid)
    rng = random.Random(311 + tid * 104729)
    rows = []
    for _ in range(n):
        k = rng.uniform(-0.15, 0.15)
        tt = rng.uniform(0.10, 2.00)
        rows.append(((k, tt), fval(k, tt, cf) + rng.gauss(0.0, sigma_noise)))
    return rows, cf


def gen_held(tid, tm):
    """Far-strike wing split with irreducible noise scaled to baseline RMSE."""
    cf = coeffs(tid)
    rng = random.Random(4241 + tid * 20261)
    pts = []
    for _ in range(N_HELD):
        # deep OTM / ITM: |k| pushed well beyond the traded band
        mag = rng.uniform(0.18, KW)
        k = mag if rng.random() < 0.5 else -mag
        tt = rng.uniform(0.10, 2.00)
        pts.append([(k, tt), fval(k, tt, cf)])
    bmse = sum((p[1] - tm) ** 2 for p in pts) / len(pts)
    beta = 0.32 + (tid - 1) * 0.020
    sh = beta * math.sqrt(bmse)
    nrng = random.Random(9173 + tid * 15485863)
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
        tid = int(header[1])
    except Exception:
        fail("bad instance header")
    if tid < 1 or tid > 10000:
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

    train, _ = gen_train(tid)
    tm = sum(y for _, y in train) / len(train)
    held = gen_held(tid, tm)

    cx = complexity(tree)

    se = 0.0
    for (k, tt), yv in held:
        env = {"k": k, "t": tt}
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
    print("heldout_MSE=%.8f baseline_MSE=%.8f complexity=%d  Ratio: %.6f"
          % (F_mse, B_mse, cx, sc / 1000.0))


if __name__ == "__main__":
    main()
