#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for band-gap-extrapolate.

- Reads the test id from <in>'s header, then re-derives the hidden materials
  law (E0, k1, k0, k4, k3) and the VISIBLE dopant family entirely from that
  id -- identical derivation to gen.py.  It also regenerates a HELD-OUT
  extrapolation split that gen.py never sees or prints:
    Group A: the SAME visible dopants, doping fraction x pushed past the
              visible window (composition extrapolation).
    Group B: BRAND-NEW dopants whose electronegativity/radius mismatch lie
              well outside anything in the training family, at x anywhere in
              an extended range (chemistry extrapolation -- the trap).
- Parses the participant's output as ONE Python-syntax arithmetic expression
  over the variables x, dEN, dR (closed-form band-gap predictor), a small
  whitelist of unary functions, +-*/** with bounded exponents, and bounded
  numeric constants.  Any syntax/name/arity/magnitude violation, or any
  non-finite value produced anywhere in the held-out rollout, scores 0.
- Score = held-out RMSE turned into a fidelity ratio against the checker's
  own internal baseline (a plain OLS straight line fit to the TRAIN x,y --
  the "band gap depends only on average composition" baseline), with a mild
  parsimony penalty for oversized expressions.
"""
import sys
import ast
import math
import random

X_MAX_VISIBLE = 0.15
SIGMA_FRAC = 0.018

TRAIN_LADDER = {
    1: (6, 5, 0.35, 0.10),
    2: (6, 6, 0.35, 0.10),
    3: (7, 6, 0.32, 0.09),
    4: (7, 6, 0.32, 0.09),
    5: (8, 6, 0.30, 0.08),
    6: (8, 7, 0.30, 0.08),
    7: (9, 7, 0.28, 0.075),
    8: (9, 7, 0.26, 0.07),
    9: (10, 7, 0.24, 0.065),
    10: (10, 8, 0.20, 0.06),
}

EPS = 0.03
LAMBDA = 0.01
NODE_BASE = 40
MAX_NODES = 90
MAX_OUT_BYTES = 20000
MAX_CONST_ABS = 1.0e6
MAX_POW_ABS = 6

ALLOWED_VARS = {"x", "dEN", "dR"}


def _safe_exp(v):
    if v > 60.0 or v < -700.0:
        raise OverflowError("exp domain")
    return math.exp(v)


def _safe_log(v):
    if v <= 0.0:
        raise ValueError("log domain")
    return math.log(v)


def _safe_sqrt(v):
    if v < 0.0:
        raise ValueError("sqrt domain")
    return math.sqrt(v)


ALLOWED_FUNCS = {
    "sin": math.sin,
    "cos": math.cos,
    "tanh": math.tanh,
    "abs": abs,
    "exp": _safe_exp,
    "log": _safe_log,
    "sqrt": _safe_sqrt,
}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden law (identical to gen.py) ----------
def hidden_params(test_id):
    rng = random.Random(500000 + 97 * test_id)
    E0 = round(rng.uniform(1.6, 3.0), 4)
    k1 = round(rng.uniform(2.0, 5.0), 4)
    k0 = round(rng.uniform(16.0, 30.0), 4)
    k4 = round(rng.uniform(3.0, 8.0), 4)
    k3 = round(rng.uniform(3.0, 7.0), 4)
    return E0, k1, k0, k4, k3


def visible_dopants(test_id):
    n_dop, pts, den_half, dr_half = TRAIN_LADDER[test_id]
    rng = random.Random(707000 + 131 * test_id)
    dopants = []
    for _ in range(n_dop):
        dEN = round(rng.uniform(-den_half, den_half), 5)
        dR = round(rng.uniform(-dr_half, dr_half), 5)
        dopants.append((dEN, dR))
    return dopants


def true_y(params, x, dEN, dR):
    E0, k1, k0, k4, k3 = params
    return E0 - k1 * x - k0 * x * x - k4 * x * (dEN ** 2) - k3 * x * dR


def heldout_points(test_id, params):
    """Group A: same dopants, x pushed past the visible window.
    Group B: new dopants far outside the training chemistry, wider x range."""
    dopants = visible_dopants(test_id)
    rng = random.Random(313000 + 271 * test_id)
    sigma = SIGMA_FRAC * params[0]
    pts = []
    for (dEN, dR) in dopants:
        for _ in range(3):
            x = rng.uniform(X_MAX_VISIBLE * 1.05, X_MAX_VISIBLE * 1.8)
            y = true_y(params, x, dEN, dR) + rng.gauss(0.0, sigma)
            pts.append((x, dEN, dR, y))
    nB = 24 + test_id
    for _ in range(nB):
        sign_en = rng.choice([-1, 1])
        sign_r = rng.choice([-1, 1])
        dEN = sign_en * rng.uniform(0.55, 1.10)
        dR = sign_r * rng.uniform(0.09, 0.20)
        x = rng.uniform(0.0, X_MAX_VISIBLE * 1.4)
        y = true_y(params, x, dEN, dR) + rng.gauss(0.0, sigma)
        pts.append((x, dEN, dR, y))
    return pts


# ---------- expression parsing / validation ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
    ast.USub, ast.UAdd,
)


def validate_expr(tree):
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return "disallowed syntax %s" % type(node).__name__
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS):
                return "disallowed call"
            if node.keywords or len(node.args) != 1:
                return "bad function arity"
        if isinstance(node, ast.Name):
            if node.id in ALLOWED_FUNCS:
                continue
            if node.id not in ALLOWED_VARS:
                return "unknown name %s" % node.id
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                return "non-numeric constant"
            v = float(node.value)
            if v != v or v in (float("inf"), float("-inf")):
                return "non-finite constant"
            if abs(v) > MAX_CONST_ABS:
                return "constant magnitude too large"
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            exp_node = node.right
            if not (isinstance(exp_node, ast.Constant) and isinstance(exp_node.value, (int, float))
                    and not isinstance(exp_node.value, bool)):
                return "exponent must be a numeric constant"
            if abs(float(exp_node.value)) > MAX_POW_ABS:
                return "exponent magnitude too large"
    return None


def count_nodes(tree):
    return sum(1 for nd in ast.walk(tree)
               if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant)))


def parse_expression(raw):
    text = raw.strip()
    if not text:
        fail("empty output")
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    err = validate_expr(tree)
    if err:
        fail(err)
    try:
        code = compile(tree, "<expr>", "eval")
    except Exception:
        fail("compile error")
    return code, count_nodes(tree)


def eval_expr(code, x, dEN, dR):
    env = dict(ALLOWED_FUNCS)
    env["x"] = x
    env["dEN"] = dEN
    env["dR"] = dR
    try:
        v = eval(code, {"__builtins__": {}}, env)
    except Exception:
        fail("evaluation error")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        fail("non-numeric result")
    v = float(v)
    if v != v or v in (float("inf"), float("-inf")):
        fail("non-finite result")
    return v


# ---------- internal baseline: plain OLS line y=a+b*x fit to TRAIN data ----------
def ols_line(xs, ys):
    n = len(xs)
    Sx = sum(xs)
    Sy = sum(ys)
    Sxx = sum(v * v for v in xs)
    Sxy = sum(u * v for u, v in zip(xs, ys))
    denom = n * Sxx - Sx * Sx
    if abs(denom) < 1e-12:
        b = 0.0
    else:
        b = (n * Sxy - Sx * Sy) / denom
    a = (Sy - b * Sx) / n
    return a, b


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            lines = fh.read().split("\n")
        header = lines[0].split()
        test_id = int(header[0])
        n_rows = int(header[1])
        xs, ys = [], []
        for i in range(n_rows):
            parts = lines[1 + i].split()
            xs.append(float(parts[1]))
            ys.append(float(parts[4]))
        if len(xs) < 2:
            fail("degenerate train set")
    except Exception:
        fail("bad instance file")

    if test_id < 1 or test_id > 100000:
        fail("bad test id")

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace")

    code, nodes = parse_expression(text)
    if nodes > MAX_NODES:
        fail("expression too large (%d nodes)" % nodes)

    params = hidden_params(test_id)
    ho = heldout_points(test_id, params)

    se_part = 0.0
    for (x, dEN, dR, y_true) in ho:
        y_hat = eval_expr(code, x, dEN, dR)
        se_part += (y_hat - y_true) ** 2
    rmse_part = math.sqrt(se_part / len(ho))

    a, b = ols_line(xs, ys)
    se_base = sum((a + b * x - y_true) ** 2 for (x, dEN, dR, y_true) in ho)
    rmse_base = math.sqrt(se_base / len(ho))

    complexity = 1.0 + LAMBDA * max(0, nodes - NODE_BASE)
    F = 1.0 / ((rmse_part + EPS) * complexity)
    B = 1.0 / (rmse_base + EPS)
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("heldout_rmse=%.6f baseline_rmse=%.6f nodes=%d  Ratio: %.6f"
          % (rmse_part, rmse_base, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
