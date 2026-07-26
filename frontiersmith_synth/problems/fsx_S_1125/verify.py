#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the allometric-growth-law extrapolation task.

- Reads n_rows and the test id from <in> (first line), then reads the logged
  training rows to get the observed train-mean size (used ONLY as the constant
  baseline).  It regenerates the hidden law's constants (alpha, beta, K0, r0)
  and the EXTREME-p held-out split entirely from the test id -- gen.py never
  prints them, and this file is never imported by a solution (it runs outside
  the solution's sandbox).
- Parses the participant's closed-form expression <out> for size S over
  {t, p} through a strict AST whitelist (rejects imports/attributes/unknown
  names, non-finite results, oversized input).
- Score (minimisation, complexity-penalised held-out MSE):
      F = heldout_MSE * (1 + LAMBDA * complexity)
      B = baseline_MSE * (1 + LAMBDA * 1)     # baseline = constant train mean
      Ratio = min(1000, 100 * B / F) / 1000
  A constant reproduces the baseline (~0.1).  Recovering the two allometric
  power laws (capacity ~ p^alpha, rate ~ (1-p)^beta) that generate the curve
  drives held-out error down toward the irreducible-noise floor, but that
  floor keeps even a strong recovery below 1.0.  A flexible low-order
  black-box surface fit that ignores the power-law shape interpolates the
  narrow interior band well yet diverges hard once p leaves it.
"""
import sys
import math
import ast
import random

LAMBDA = 0.0002
TMAX = 10
HELD_PS = [0.08, 0.13, 0.18, 0.82, 0.87, 0.92]
ALLOWED_FUNCS = {"exp": math.exp, "log": math.log, "sin": math.sin,
                 "cos": math.cos, "sqrt": math.sqrt, "tanh": math.tanh,
                 "abs": abs}
ALLOWED_VARS = {"t", "p"}
MAX_EXPR_BYTES = 200000


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---- hidden law (identical formulas to gen.py) ----
def params(tid):
    alpha = max(0.12, 0.90 - 0.075 * tid)
    beta = max(0.12, 0.85 - 0.065 * tid)
    K0 = 10.0
    r0 = 0.15
    sigma = 0.10 + 0.03 * tid
    return alpha, beta, K0, r0, sigma


def true_S(t, p, alpha, beta, K0, r0):
    K = K0 * (p ** alpha)
    r = r0 * ((1.0 - p) ** beta)
    return K * (1.0 - math.exp(-r * t))


def gen_held(tid, alpha, beta, K0, r0, train_mean):
    """Extreme-p split: p far outside the [0.42,0.58] training band, with
    irreducible noise scaled to the baseline RMSE (like gentle-vs-violent
    template problems) so a perfect law recovery still cannot reach Ratio 1."""
    pts = []
    for p in HELD_PS:
        for t in range(1, TMAX + 1):
            pts.append([t, p, true_S(t, p, alpha, beta, K0, r0)])
    bmse = sum((s - train_mean) ** 2 for _, _, s in pts) / len(pts)
    frac = 0.35 + (tid - 1) * 0.02
    sh = frac * math.sqrt(bmse)
    nrng = random.Random(555001 + tid * 91301)
    for pt in pts:
        pt[2] = pt[2] + nrng.gauss(0.0, sh)
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
            lines = fh.read().splitlines()
        header = lines[0].split()
        n_rows = int(header[0])
        tid = int(header[1])
        s_vals = []
        for ln in lines[1:1 + n_rows]:
            parts = ln.split()
            s_vals.append(float(parts[2]))
        if not s_vals:
            raise ValueError("no rows")
        train_mean = sum(s_vals) / len(s_vals)
    except Exception:
        fail("bad instance header/body")
    if tid < 1 or tid > 100000:
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
    lines2 = [ln for ln in expr.splitlines() if ln.strip()]
    if len(lines2) != 1:
        fail("expression must be a single line")
    expr = lines2[0].strip()

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

    # hidden ground truth + held-out extrapolation split
    alpha, beta, K0, r0, _sigma = params(tid)
    held = gen_held(tid, alpha, beta, K0, r0, train_mean)

    cx = complexity(tree)

    se = 0.0
    for t, p, yv in held:
        env = {"t": float(t), "p": float(p)}
        env.update(ALLOWED_FUNCS)
        try:
            pred = eval(code, {"__builtins__": {}}, env)
        except Exception:
            fail("evaluation error")
        if isinstance(pred, bool) or not isinstance(pred, (int, float)):
            fail("non-numeric result")
        pred = float(pred)
        if pred != pred or pred in (float("inf"), float("-inf")):
            fail("non-finite result")
        d = pred - yv
        se += d * d
    F_mse = se / len(held)

    B_mse = sum((yv - train_mean) ** 2 for _, _, yv in held) / len(held)
    B = B_mse * (1.0 + LAMBDA * 1)
    F = F_mse * (1.0 + LAMBDA * cx)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("heldout_MSE=%.6f baseline_MSE=%.6f complexity=%d  Ratio: %.6f"
          % (F_mse, B_mse, cx, sc / 1000.0))


if __name__ == "__main__":
    main()
