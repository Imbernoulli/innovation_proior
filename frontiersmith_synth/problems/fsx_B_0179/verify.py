#!/usr/bin/env python3
"""
Deterministic checker for the e-sports "clutch-rating" symbolic-regression task.

Usage:  python3 verify.py <in> <out> <ans>      (ans is an ignored placeholder)

The participant's <out> is a single closed-form expression string over the
variables x1,x2,x3,x4.  We:
  1. parse it with a strict AST whitelist (any illegal node / name -> Ratio 0.0);
  2. evaluate it on a HELD-OUT EXTRAPOLATION set regenerated deterministically
     here (region [1.0,1.8]^4, disjoint from the train region [0,1]^4 the solver
     saw) -- rejecting any non-finite value;
  3. score it from held-out normalized error AND an expression-complexity penalty;
  4. normalise against the checker's own trivial baseline (predict the train mean)
     so that reproducing the baseline scores ~0.1 and better generalisation climbs.

Deterministic: fixed seed, pure-python arithmetic, no wall-time / randomness.
"""
import sys
import ast
import math
import random

ALLOWED_VARS = {"x1", "x2", "x3", "x4"}
# name -> (callable, domain_guard(returns bool ok))
ALLOWED_FUNCS = {
    "exp":  (math.exp,  lambda a: True),
    "log":  (math.log,  lambda a: a > 0.0),
    "sqrt": (math.sqrt, lambda a: a >= 0.0),
    "sin":  (math.sin,  lambda a: True),
    "cos":  (math.cos,  lambda a: True),
    "tanh": (math.tanh, lambda a: True),
    "abs":  (abs,       lambda a: True),
}
MAX_EXPR_LEN = 8000
MAX_ABS_POW = 6.0          # constant exponent magnitude cap
GAMMA = 0.02               # complexity penalty strength
HELD_SEED = 424242
HELD_N = 200
HELD_SIGMA = 0.30          # irreducible held-out noise -> caps achievable fit


def reject(reason):
    print("infeasible: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


# ---------------------------------------------------------------- ground truth
def f_clean(x1, x2, x3, x4):
    return (1.5
            + 2.2 * x1 ** 2
            - 1.1 * x2
            + 1.4 * math.exp(0.6 * x3)
            + 0.9 * x1 * x4
            - 0.7 * x2 * x3
            + 0.4 * x4 ** 2)


def make_heldout():
    rng = random.Random(HELD_SEED)
    X = []
    Y = []
    for _ in range(HELD_N):
        x = [rng.uniform(1.0, 1.8) for _ in range(4)]   # EXTRAPOLATION region
        y = f_clean(*x) + rng.gauss(0.0, HELD_SIGMA)
        X.append(x)
        Y.append(y)
    return X, Y


# ---------------------------------------------------------------- expr parsing
def validate_and_count(node):
    """Recursively validate the AST against the whitelist and return node count."""
    if isinstance(node, ast.Expression):
        return validate_and_count(node.body)
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
            reject("non-numeric constant")
        return 1
    if isinstance(node, ast.Name):
        if node.id not in ALLOWED_VARS:
            reject("illegal name '%s'" % node.id)
        return 1
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            reject("illegal binary operator")
        if isinstance(node.op, ast.Pow):
            # exponent must be a (possibly signed) numeric constant, bounded
            e = node.right
            val = None
            if isinstance(e, ast.Constant) and isinstance(e.value, (int, float)):
                val = float(e.value)
            elif (isinstance(e, ast.UnaryOp) and isinstance(e.op, (ast.USub, ast.UAdd))
                  and isinstance(e.operand, ast.Constant)
                  and isinstance(e.operand.value, (int, float))):
                base = float(e.operand.value)
                val = -base if isinstance(e.op, ast.USub) else base
            if val is None:
                reject("non-constant exponent")
            if abs(val) > MAX_ABS_POW:
                reject("exponent magnitude too large")
        return 1 + validate_and_count(node.left) + validate_and_count(node.right)
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.USub, ast.UAdd)):
            reject("illegal unary operator")
        return 1 + validate_and_count(node.operand)
    if isinstance(node, ast.Call):
        if node.keywords or not isinstance(node.func, ast.Name):
            reject("illegal call form")
        if node.func.id not in ALLOWED_FUNCS:
            reject("illegal function '%s'" % node.func.id)
        if len(node.args) != 1:
            reject("function arity")
        return 1 + validate_and_count(node.args[0])
    reject("illegal syntax node %s" % type(node).__name__)


def compile_expr(expr):
    if len(expr) > MAX_EXPR_LEN:
        reject("expression too long")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        reject("parse error")
    complexity = validate_and_count(tree)
    code = compile(tree, "<expr>", "eval")
    return code, complexity


def eval_point(code, x):
    env = {"__builtins__": {}, "x1": x[0], "x2": x[1], "x3": x[2], "x4": x[3]}
    # bind guarded functions
    def mk(fn, guard):
        def g(a):
            a = float(a)
            if not guard(a):
                raise ValueError("domain")
            return fn(a)
        return g
    for name, (fn, guard) in ALLOWED_FUNCS.items():
        env[name] = mk(fn, guard)
    try:
        v = eval(code, env, {})
    except Exception:
        return None
    try:
        v = float(v)
    except Exception:
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v


# ---------------------------------------------------------------- scoring
def qeff(code, complexity, Xho, Yho):
    """Effective quality in [0, ~1): held-out fit shrunk by a complexity penalty.
    Returns None if the expression is infeasible on the held-out set."""
    preds = []
    for x in Xho:
        v = eval_point(code, x)
        if v is None:
            return None
        preds.append(v)
    n = len(Yho)
    mean = sum(Yho) / n
    var = sum((y - mean) ** 2 for y in Yho) / n
    if var <= 0.0:
        var = 1e-9
    mse = sum((p - y) ** 2 for p, y in zip(preds, Yho)) / n
    nmse = mse / var
    q = 1.0 / (1.0 + nmse)
    return q / (1.0 + GAMMA * complexity)


def read_train_mean(inpath):
    with open(inpath) as f:
        first = f.readline().split()
        if len(first) < 2:
            return 0.0
        n = int(first[0])
        ys = []
        for _ in range(n):
            parts = f.readline().split()
            if len(parts) < 5:
                break
            ys.append(float(parts[4]))
    if not ys:
        return 0.0
    return sum(ys) / len(ys)


def read_submission(outpath):
    try:
        with open(outpath) as f:
            data = f.read(MAX_EXPR_LEN + 64)
    except Exception:
        reject("cannot read output")
    for line in data.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        return s
    reject("empty output")


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0")
        sys.exit(0)
    inpath, outpath = sys.argv[1], sys.argv[2]

    Xho, Yho = make_heldout()

    # --- checker's own trivial baseline: predict the train mean (a bare constant)
    tmean = read_train_mean(inpath)
    base_code, base_c = compile_expr(repr(float(tmean)))
    base_q = qeff(base_code, base_c, Xho, Yho)
    if base_q is None or base_q <= 0.0:
        base_q = 1e-9

    # --- participant expression
    expr = read_submission(outpath)
    code, complexity = compile_expr(expr)
    sub_q = qeff(code, complexity, Xho, Yho)
    if sub_q is None:
        reject("expression non-finite / out of domain on held-out set")

    sc = min(1000.0, 100.0 * sub_q / max(1e-9, base_q))
    print("held_out_qeff=%.6f baseline_qeff=%.6f complexity=%d"
          % (sub_q, base_q, complexity))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
