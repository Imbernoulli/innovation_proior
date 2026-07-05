#!/usr/bin/env python3
"""
Deterministic scorer for fsx_B_0171 (format E, symbolic regression).

The participant reads the TRAIN sample from stdin and writes ONE closed-form
expression over x0,x1,x2,x3 to stdout. This checker:
  1. parses the expression with a strict AST whitelist (arithmetic + a short list
     of unary functions; only variables x0..x3 and constants pi,e allowed),
  2. regenerates the HELD-OUT EXTRAPOLATION split deterministically (a hidden
     region OUTSIDE the training box -> rewards generalization, not memorization),
  3. evaluates the expression there, rejecting any non-finite output,
  4. scores from held-out RMSE plus a complexity penalty, normalized against an
     internal constant-predictor baseline B.

Objective is a LOSS (lower is better): F = RMSE_heldout + ALPHA * complexity.
Baseline B = RMSE of the constant predictor (= mean of train targets).
  sc = min(1000, 100 * B / max(1e-9, F));  print Ratio: sc/1000.
Trivial constant -> ~0.1; recovering the law -> high but < 1 (irreducible noise).
"""
import sys
import ast
import numpy as np

ALPHA = 0.004            # complexity penalty per AST node
MAX_CHARS = 5000
MAX_NODES = 400

ALLOWED_FUNCS = {"exp", "log", "sqrt", "sin", "cos", "tanh", "abs"}
ALLOWED_VARS = {"x0", "x1", "x2", "x3"}
ALLOWED_CONSTS = {"pi", "e"}
ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.USub, ast.UAdd,
)


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


# ---------- hidden ground truth (mirrors gen.py) ----------
def _law(X):
    x0, x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return 5.0 * x0 * np.exp(-1.4 * x1) + 2.0 * x2**2 + 1.5 * x3**2 - 0.8 * x0 * x3 + 1.5


def _heldout():
    # EXTRAPOLATION region: strictly outside the training box [0,1]^4-ish.
    rng = np.random.default_rng(424242)
    N = 500
    x0 = rng.uniform(1.20, 1.90, N)
    x1 = rng.uniform(1.20, 1.90, N)
    x2 = rng.uniform(1.20, 1.80, N)
    x3 = rng.uniform(1.20, 1.80, N)
    X = np.stack([x0, x1, x2, x3], axis=1)
    y = _law(X) + rng.normal(0.0, 0.45, N)          # irreducible held-out noise -> headroom
    return X, y


# ---------- strict AST validation ----------
def validate_ast(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            return False, "node %s not allowed" % type(node).__name__
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCS:
                return False, "bad function call"
            if node.keywords or len(node.args) != 1:
                return False, "functions take exactly one arg"
        if isinstance(node, ast.Name):
            if node.id not in ALLOWED_VARS and node.id not in ALLOWED_CONSTS and node.id not in ALLOWED_FUNCS:
                return False, "unknown name %r" % node.id
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            return False, "non-numeric constant"
    return True, ""


def make_env(X):
    return {
        "__builtins__": {},
        "x0": X[:, 0], "x1": X[:, 1], "x2": X[:, 2], "x3": X[:, 3],
        "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "sin": np.sin,
        "cos": np.cos, "tanh": np.tanh, "abs": np.abs,
        "pi": np.pi, "e": np.e,
    }


def main():
    # ----- read training instance (for the baseline mean) -----
    try:
        toks = open(sys.argv[1]).read().split()
        it = iter(toks)
        n = int(next(it))
        ys = []
        for _ in range(n):
            next(it); next(it); next(it); next(it)      # x0..x3
            ys.append(float(next(it)))
        ytr = np.array(ys, dtype=float)
    except Exception:
        fail("bad input")

    # ----- read participant expression -----
    raw = open(sys.argv[2]).read()
    expr = raw.strip()
    if not expr:
        fail("empty output")
    if len(expr) > MAX_CHARS:
        fail("expression too long")
    if "\n" in expr:
        fail("expression must be a single line")

    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        fail("parse error")

    okv, why = validate_ast(tree)
    if not okv:
        fail(why)

    complexity = sum(1 for _ in ast.walk(tree))
    if complexity > MAX_NODES:
        fail("too complex (%d nodes)" % complexity)

    # ----- regenerate held-out extrapolation split & evaluate -----
    Xho, yho = _heldout()
    try:
        code = compile(tree, "<expr>", "eval")
        with np.errstate(all="ignore"):
            pred = eval(code, make_env(Xho))
        pred = np.asarray(pred, dtype=float)
        if pred.ndim == 0:                              # a constant expression
            pred = np.full(yho.shape, float(pred))
        if pred.shape != yho.shape:
            fail("prediction shape mismatch")
    except Exception:
        fail("evaluation error")

    if not np.all(np.isfinite(pred)):
        fail("non-finite prediction")

    rmse = float(np.sqrt(np.mean((pred - yho) ** 2)))
    F = rmse + ALPHA * complexity

    # ----- internal baseline: constant predictor = mean of train targets -----
    B = float(np.sqrt(np.mean((np.full(yho.shape, float(ytr.mean())) - yho) ** 2)))
    B = max(1e-9, B)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("rmse=%.5f complexity=%d F=%.5f B=%.5f Ratio: %.6f" %
          (rmse, complexity, F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
