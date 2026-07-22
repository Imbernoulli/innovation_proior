#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the alien-billiards collision-invariant-mining task.

- Reads the test id from <in> (first line), then regenerates the hidden
  constants (mass ratio r, velocity ceiling c), the GENTLE training logbook
  (only for the constant baseline) and the VIOLENT high-energy HELD-OUT split
  entirely from that id.  The physics + constants live ONLY here -- gen.py never
  prints them.
- Parses the participant's closed-form expression <out> for v1 over {u1, u2}
  through a strict AST whitelist (rejects imports/attributes/unknown names,
  non-finite results, oversized input).
- Score (minimisation, complexity-penalised held-out MSE):
      F = heldout_MSE * (1 + LAMBDA * complexity)
      B = baseline_MSE * (1 + LAMBDA * 1)       # baseline = constant train mean
      Ratio = min(1000, 100 * B / F) / 1000
  A constant reproduces the baseline (~0.1).  Extrapolating the collision map to
  the ceiling regime drives the held-out error down toward the irreducible-noise
  floor and raises the ratio, but that floor plus the hidden ceiling keep even a
  strong recovery below 1.0; a low-energy black-box polynomial fit that ignores
  the invariants over-shoots the ceiling and stays far below the strong recovery.
"""
import sys
import math
import ast
import random

LAMBDA = 0.0002
N_HELD = 250
ALLOWED_FUNCS = {"exp": math.exp, "log": math.log, "sin": math.sin,
                 "cos": math.cos, "sqrt": math.sqrt, "tanh": math.tanh,
                 "abs": abs}
ALLOWED_VARS = {"u1", "u2"}
MAX_EXPR_BYTES = 200000


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---- hidden physics (identical to gen.py) ----
def params(t):
    rng = random.Random(770001 + t * 5303)
    if t % 2 == 0:
        r = rng.uniform(1.70, 2.90)
    else:
        r = rng.uniform(0.30, 0.62)
    c = rng.uniform(1.60, 2.60)
    return r, c


def gamma(u, c):
    return 1.0 / math.sqrt(abs(1.0 - (u / c) ** 2))


def post(u1, u2, r, c):
    g1 = gamma(u1, c)
    g2 = gamma(u2, c)
    p = g1 * u1 + g2 * r * u2
    E = g1 + g2 * r
    V = p / E
    c2 = c * c
    w1 = (u1 - V) / (1.0 - u1 * V / c2)
    v1 = (V - w1) / (1.0 - w1 * V / c2)
    w2 = (u2 - V) / (1.0 - u2 * V / c2)
    v2 = (V - w2) / (1.0 - w2 * V / c2)
    return v1, v2


def gen_train(t):
    r, c = params(t)
    sigma = 0.008 + (t - 1) * 0.004
    n = 190 - (t - 1) * 12
    rng = random.Random(4200 + t * 99131)
    rows = []
    for _ in range(n):
        s1 = rng.uniform(-0.55, 0.55)
        s2 = rng.uniform(-0.55, 0.55)
        u1 = c * s1
        u2 = c * s2
        v1, v2 = post(u1, u2, r, c)
        rows.append(v1 + rng.gauss(0.0, sigma))  # only v1 needed for the mean
    return rows


def gen_held(t, tm):
    """Violent high-energy split: speed fraction |s| in [0.70, 0.93] of the
    ceiling (the training logbook never exceeds 0.55), with irreducible noise
    scaled to the baseline RMSE so a perfect map still cannot reach Ratio ~ 1."""
    r, c = params(t)
    rng = random.Random(9100 + t * 30011)
    pts = []
    for _ in range(N_HELD):
        a1 = rng.uniform(0.70, 0.93)
        a2 = rng.uniform(0.70, 0.93)
        u1 = c * a1 * (1 if rng.random() < 0.5 else -1)
        u2 = c * a2 * (1 if rng.random() < 0.5 else -1)
        v1, _ = post(u1, u2, r, c)
        pts.append([u1, u2, v1])
    bmse = sum((p[2] - tm) ** 2 for p in pts) / len(pts)
    beta = 0.375 + (t - 1) * 0.004
    sh = beta * math.sqrt(bmse)
    nrng = random.Random(13000 + t * 7778777)
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
    if t < 1 or t > 100000:
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

    # hidden ground truth + splits
    train_v1 = gen_train(t)
    tm = sum(train_v1) / len(train_v1)
    held = gen_held(t, tm)

    cx = complexity(tree)

    se = 0.0
    for x1, x2, yv in held:
        env = {"u1": x1, "u2": x2}
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
