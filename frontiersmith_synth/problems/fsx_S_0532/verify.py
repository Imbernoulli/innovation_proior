#!/usr/bin/env python3
"""
Deterministic scorer for fsx_S_0532 (format E, family hidden-affine-sparse-poly).

The participant reads the TRAIN sample (noisy (x, y) rows inside a narrow window)
from stdin and writes ONE closed-form expression in the variable `x` to stdout.
This checker:
  1. parses the expression under a strict AST whitelist (arithmetic + a short list
     of unary functions; only the variable `x` and constants pi, e allowed),
  2. regenerates the HELD-OUT EXTRAPOLATION split deterministically -- raw readings
     in a region OUTSIDE the calibration window (rewards generalization, not
     memorization of the window),
  3. evaluates the expression there, rejecting any non-finite output,
  4. scores from held-out RMSE plus a small complexity penalty, normalized against
     an internal constant-predictor baseline B.

Objective is a LOSS (lower is better): F = RMSE_heldout + ALPHA * complexity.
Baseline B = held-out RMSE of the constant predictor (= mean of train targets).
  sc = min(1000, 100 * B / max(1e-9, F));  print Ratio: sc/1000.
Reproducing the constant baseline -> Ratio ~ 0.1. Recovering the hidden affine +
sparse law drives the held-out RMSE down and pushes the ratio up, but irreducible
held-out noise + extrapolation-amplified coefficient error keep it well below 1.0.
"""
import sys
import ast
import numpy as np

ALPHA = 0.002            # complexity penalty per AST node
MAX_CHARS = 5000
MAX_NODES = 400

ALLOWED_FUNCS = {"exp", "log", "sqrt", "sin", "cos", "tanh", "abs"}
ALLOWED_VARS = {"x"}
ALLOWED_CONSTS = {"pi", "e"}
ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.USub, ast.UAdd,
)


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


# ---------- hidden ground truth (mirrors gen.py byte-for-byte) ----------
A_TRUE, B_TRUE = 2.0, -1.0
TERMS = [(1.5, 4), (-2.0, 3), (1.0, 1)]


def _law(x):
    u = A_TRUE * x + B_TRUE
    return sum(c * u ** e for c, e in TERMS)


def _heldout():
    # EXTRAPOLATION region: raw readings strictly OUTSIDE the [0,0.9] window.
    rng = np.random.default_rng(20250532)
    N = 400
    x = rng.uniform(1.10, 1.75, N)
    y = _law(x) + rng.normal(0.0, 1.50, N)          # irreducible held-out noise -> headroom
    return x, y


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


def make_env(x):
    return {
        "__builtins__": {},
        "x": x,
        "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "sin": np.sin,
        "cos": np.cos, "tanh": np.tanh, "abs": np.abs,
        "pi": np.pi, "e": np.e,
    }


def main():
    # ----- read training instance (for the constant baseline mean) -----
    try:
        toks = open(sys.argv[1]).read().split()
        it = iter(toks)
        n = int(next(it))
        ys = []
        for _ in range(n):
            next(it)                                  # x
            ys.append(float(next(it)))                # y
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
        if pred.ndim == 0:                            # a constant expression
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
