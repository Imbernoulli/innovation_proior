#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the tidal-gauge Fourier-sparse recovery task.

- Reads the test id from <in> (first line), then regenerates the hidden tidal
  law, the TRAIN window and the FUTURE (held-out extrapolation) window entirely
  from that id.  The law lives ONLY here -- never in gen.py's stdout.
- Parses the participant's closed-form expression in the single variable `t`
  through a strict AST whitelist (rejects imports/attributes/unknown names,
  non-finite results, oversized input).
- Score (minimisation, complexity-penalised held-out MSE):
      F = heldout_MSE * (1 + LAMBDA * complexity)
      B = baseline_MSE * (1 + LAMBDA * 1)      # baseline = constant train mean
      Ratio = min(1000, 100*B/F) / 1000
  Predicting the train mean reproduces the baseline (~0.1).  Recovering the
  hidden constituents drives the future-window error toward the irreducible
  instrument-noise floor and raises the ratio, but the noise floor plus the
  fact that a short window cannot perfectly resolve the off-grid frequencies
  keep even a strong recovery below 1.0.
"""
import sys, math, ast, random

TWO_PI = 2.0 * math.pi
LAMBDA = 0.002
L_HELD = 72                       # future-window length in hours
GAP = 1                          # hours between train end and held start
ALLOWED_FUNCS = {"exp": math.exp, "log": math.log, "sin": math.sin,
                 "cos": math.cos, "sqrt": math.sqrt, "tanh": math.tanh,
                 "abs": abs}
ALLOWED_VARS = {"t"}
MAX_EXPR_BYTES = 200000
BASE_PERIODS = [6.0, 9.0, 13.0, 19.0, 31.0]


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---- hidden ground truth (identical construction to gen.py) ----
def n_constituents(t):
    return 2 if t <= 3 else (3 if t <= 6 else 4)


def constituents(t):
    rng = random.Random(0xC0FFEE + t * 7919)
    K = n_constituents(t)
    mu = rng.uniform(0.5, 2.0)
    chosen = rng.sample(BASE_PERIODS, K)
    cons = []
    for P0 in chosen:
        P = P0 * rng.uniform(0.97, 1.03)
        omega = TWO_PI / P
        A = rng.uniform(0.3, 1.5)
        phi = rng.uniform(0.0, TWO_PI)
        cons.append((omega, A, phi))
    return mu, cons


def height(tval, mu, cons):
    h = mu
    for omega, A, phi in cons:
        h += A * math.cos(omega * tval + phi)
    return h


def train_window(t):
    L = 168 - (t - 1) * 11
    return L + 1, L


def gen_train_mean(t):
    n, L = train_window(t)
    mu, cons = constituents(t)
    sigma = 0.03 + (t - 1) * 0.012
    rng = random.Random(500 + t * 104729)
    s = 0.0
    for i in range(n):
        s += height(float(i), mu, cons) + rng.gauss(0.0, sigma)
    return s / n


def gen_held(t, tm):
    """Future extrapolation window with irreducible instrument noise scaled to
    the baseline (mean-predictor) RMSE, so a noise floor bounds the score."""
    n, L = train_window(t)
    mu, cons = constituents(t)
    start = L + GAP
    pts = []
    for i in range(L_HELD):
        tv = float(start + i)
        pts.append([tv, height(tv, mu, cons)])
    bmse = sum((p[1] - tm) ** 2 for p in pts) / len(pts)
    beta = 0.42 + (t - 1) * 0.01
    sh = beta * math.sqrt(max(1e-12, bmse))
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

    tm = gen_train_mean(t)
    held = gen_held(t, tm)
    cx = complexity(tree)

    se = 0.0
    for tv, yv in held:
        env = {"t": tv}
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
