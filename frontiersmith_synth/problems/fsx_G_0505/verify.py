#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic checker for the battery-lab aging-law symbolic regression task.
The participant sees early-cycle, low-temperature training rows and emits one
closed-form expression over N, T, D, R.  This checker regenerates a high-cycle,
high-temperature held-out split from the test id and scores held-out MSE with a
small expression-complexity penalty.
"""
import ast
import math
import random
import sys


LAMBDA = 0.0025
N_HELD = 340
MAX_EXPR_BYTES = 200000
MAX_COMPLEXITY = 700
MAX_ABS_PRED = 1.0e6

ALLOWED_FUNCS = {
    "exp": math.exp,
    "log": math.log,
    "sin": math.sin,
    "cos": math.cos,
    "sqrt": math.sqrt,
    "tanh": math.tanh,
    "abs": abs,
}
ALLOWED_VARS = {"N", "T", "D", "R"}
ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.USub, ast.UAdd,
)


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


def coeffs(t):
    rng = random.Random(90731 + 104729 * t)
    return (
        rng.uniform(0.010, 0.040),
        rng.uniform(0.070, 0.150),
        rng.uniform(0.46, 0.62),
        rng.uniform(0.20, 0.36),
        rng.uniform(0.012, 0.040),
        rng.uniform(0.34, 0.58),
        rng.uniform(0.010, 0.030),
        rng.uniform(1.25, 1.55),
        rng.uniform(0.45, 0.78),
        rng.uniform(1.45, 2.15),
        rng.uniform(0.003, 0.010),
    )


def softplus(z):
    if z > 45.0:
        return z
    if z < -45.0:
        return math.exp(z)
    return math.log1p(math.exp(z))


def fval(N, T, D, R, cf):
    q0, A, alpha, bt, B, bo, C, gamma, bh, knee, H = cf
    u = N / 100.0
    theta = (T - 20.0) / 12.0
    stress = D * R
    act = 0.55 * softplus((u - knee) / 0.55)
    sei = A * (u ** alpha) * math.exp(bt * theta) * (D ** 1.05)
    throughput = B * u * math.exp(bo * theta) * stress
    hot_knee = C * (act ** gamma) * math.exp(bh * theta) * (stress ** 1.15)
    cross = H * (u ** 0.80) * math.exp(0.35 * theta) * D * (0.25 + R) * (0.8 + 0.4 * theta)
    return q0 + sei + throughput + hot_knee + cross


def gen_train(t):
    n = 210 - 10 * (t - 1)
    sigma = 0.010 + 0.005 * t
    cf = coeffs(t)
    rng = random.Random(55621 + 65537 * t)
    rows = []
    for _ in range(n):
        if rng.random() < 0.72:
            N = rng.uniform(25.0, 180.0)
        else:
            N = rng.uniform(180.0, 320.0)
        T = rng.uniform(6.0, 24.0)
        D = rng.uniform(0.20, 0.72)
        R = rng.uniform(0.25, 1.15)
        y = fval(N, T, D, R, cf) + rng.gauss(0.0, sigma)
        rows.append((N, T, D, R, y))
    return rows


def gen_held(t, train_mean):
    cf = coeffs(t)
    rng = random.Random(84191 + 20261 * t)
    pts = []
    for _ in range(N_HELD):
        N = rng.uniform(650.0, 1700.0)
        T = rng.uniform(34.0, 58.0)
        D = rng.uniform(0.45, 0.95)
        R = rng.uniform(0.70, 2.10)
        pts.append([N, T, D, R, fval(N, T, D, R, cf)])

    bmse = sum((p[4] - train_mean) ** 2 for p in pts) / len(pts)
    frac = 0.42 + 0.018 * (t - 1)
    sigma = frac * math.sqrt(max(1e-12, bmse))
    nrng = random.Random(990001 + 15485863 * t)
    for p in pts:
        p[4] += nrng.gauss(0.0, sigma)
    return pts


def validate_ast(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
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
    return sum(
        1 for nd in ast.walk(tree)
        if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant))
    )


def read_test_id(inf):
    try:
        with open(inf) as fh:
            parts = fh.readline().split()
        if len(parts) != 2:
            fail("bad instance header")
        t = int(parts[1])
    except Exception:
        fail("bad instance header")
    if not (1 <= t <= 10):
        fail("bad test id")
    return t


def read_expression(outf):
    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_EXPR_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_EXPR_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace").strip()
    if not text:
        fail("empty expression")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) != 1:
        fail("expression must be a single line")
    return lines[0].strip()


def main():
    if len(sys.argv) < 3:
        fail("usage")
    t = read_test_id(sys.argv[1])
    expr = read_expression(sys.argv[2])

    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        fail("parse error")
    reason = validate_ast(tree)
    if reason:
        fail(reason)
    cx = complexity(tree)
    if cx > MAX_COMPLEXITY:
        fail("expression too complex")
    try:
        code = compile(tree, "<expr>", "eval")
    except Exception:
        fail("compile error")

    train = gen_train(t)
    train_mean = sum(row[4] for row in train) / len(train)
    held = gen_held(t, train_mean)

    se = 0.0
    for N, T, D, R, y in held:
        env = {"N": N, "T": T, "D": D, "R": R}
        env.update(ALLOWED_FUNCS)
        try:
            pred = eval(code, {"__builtins__": {}}, env)
        except Exception:
            fail("evaluation error")
        if not isinstance(pred, (int, float)) or isinstance(pred, bool):
            fail("non-numeric result")
        pred = float(pred)
        if not math.isfinite(pred):
            fail("non-finite result")
        if abs(pred) > MAX_ABS_PRED:
            fail("prediction out of range")
        diff = pred - y
        se += diff * diff
        if not math.isfinite(se):
            fail("non-finite objective")

    held_mse = se / len(held)
    base_mse = sum((p[4] - train_mean) ** 2 for p in held) / len(held)
    F = held_mse * (1.0 + LAMBDA * cx)
    B = base_mse * (1.0 + LAMBDA)
    if not (math.isfinite(F) and F > 0.0 and math.isfinite(B) and B > 0.0):
        fail("bad objective")
    sc = min(1000.0, 100.0 * B / max(1e-12, F))
    print(
        "heldout_MSE=%.9g baseline_MSE=%.9g complexity=%d  Ratio: %.6f"
        % (held_mse, base_mse, cx, sc / 1000.0)
    )


if __name__ == "__main__":
    main()
