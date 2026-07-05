#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic scorer for the grid-efficiency scaling-law EXTRAPOLATION problem.

- <in>  : the TRAIN rows the solver saw (regenerated identically here anyway).
- <out> : the solver's submitted CLOSED-FORM expression in the variable x.

The checker regenerates the hidden law + a held-out LARGE-SCALE region
(genuine extrapolation), evaluates the submitted expression there, and scores
extrapolation RMSE (with a small complexity penalty) against an internal
constant baseline B.  Minimization objective:

    sc = min(1000, 100 * B / max(eps, eff_error));  Ratio = sc/1000

Any feasibility violation (unparseable / disallowed / non-finite / absurd
output) prints  Ratio: 0.0  and exits 0.
"""
import sys
import ast
import math


# ============ hidden ground truth (mirrors gen.py exactly) ============
def _rng(seed):
    state = [(seed * 2654435761 + 12345) & 0x7FFFFFFF]

    def nxt():
        state[0] = (1103515245 * state[0] + 12345) & 0x7FFFFFFF
        return state[0] / 0x7FFFFFFF

    return nxt


def derive_params(test_id):
    r = _rng(1000 + test_id)
    c = 0.90 + 0.08 * r()
    a = 0.30 + 0.40 * r()
    b = 0.35 + 0.30 * r()
    return c, a, b


def noise_rel(test_id):
    return 0.012 + 0.004 * (test_id - 1)


def train_x(n=24, lo=8.0, hi=250.0):
    return [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)]


def held_x(n=14, lo=400.0, hi=4000.0):
    # EXTRAPOLATION region: strictly larger scales than any train point
    return [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)]


def make_train_y(test_id):
    c, a, b = derive_params(test_id)
    nr = noise_rel(test_id)
    rn = _rng(5000 + test_id)
    ys = []
    for x in train_x():
        clean = c - a * x ** (-b)
        ys.append(clean * (1.0 + nr * (2.0 * rn() - 1.0)))
    return ys


def make_held(test_id):
    c, a, b = derive_params(test_id)
    nr = noise_rel(test_id)
    rn = _rng(9000 + test_id)   # separate held-out noise stream
    xs = held_x()
    ys = []
    for x in xs:
        clean = c - a * x ** (-b)
        ys.append(clean * (1.0 + nr * (2.0 * rn() - 1.0)))
    return xs, ys


# ============ safe expression evaluation (whitelist AST) ============
_ALLOWED_FUNCS = {
    "log": math.log, "exp": math.exp, "sqrt": math.sqrt,
    "sin": math.sin, "cos": math.cos, "tanh": math.tanh,
    "abs": abs, "log10": math.log10, "pow": pow,
}
_ALLOWED_CONSTS = {"pi": math.pi, "e": math.e}


def _check_node(node):
    """Raise ValueError on any disallowed construct."""
    if isinstance(node, ast.Expression):
        _check_node(node.body)
    elif isinstance(node, ast.BinOp):
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult,
                                    ast.Div, ast.Pow, ast.Mod)):
            raise ValueError("bad binop")
        _check_node(node.left)
        _check_node(node.right)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise ValueError("bad unaryop")
        _check_node(node.operand)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise ValueError("bad call")
        if node.keywords:
            raise ValueError("kwargs")
        for a in node.args:
            _check_node(a)
    elif isinstance(node, ast.Name):
        if node.id != "x" and node.id not in _ALLOWED_CONSTS:
            raise ValueError("bad name: %s" % node.id)
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
            raise ValueError("bad const")
        if not math.isfinite(float(node.value)):
            raise ValueError("nonfinite const")
    elif isinstance(node, ast.Num):  # legacy py
        if not math.isfinite(float(node.n)):
            raise ValueError("nonfinite const")
    else:
        raise ValueError("disallowed node: %s" % type(node).__name__)


def _count_nodes(node):
    return sum(1 for _ in ast.walk(node))


def compile_expr(text):
    """Return (evalfn, complexity) or raise ValueError."""
    if len(text) > 4000:
        raise ValueError("expr too long")
    tree = ast.parse(text, mode="eval")
    _check_node(tree)
    complexity = _count_nodes(tree.body)
    code = compile(tree, "<expr>", "eval")

    def ev(xval):
        env = {"x": xval}
        env.update(_ALLOWED_CONSTS)
        env.update(_ALLOWED_FUNCS)
        return eval(code, {"__builtins__": {}}, env)

    return ev, complexity


def fail(reason):
    print("reason: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def rmse(pred, targ):
    s = 0.0
    for p, t in zip(pred, targ):
        d = p - t
        s += d * d
    return math.sqrt(s / len(targ))


def main():
    if len(sys.argv) < 3:
        fail("usage")
    in_path, out_path = sys.argv[1], sys.argv[2]

    # infer test_id from train rows: match the deterministic train x-grid & y
    # (the harness names files by case, but we reconstruct from data content).
    try:
        raw_in = open(in_path).read().split()
        train_pairs = []
        it = iter(raw_in)
        for a in it:
            b = next(it)
            train_pairs.append((float(a), float(b)))
    except Exception:
        fail("bad instance")

    # identify test_id by matching regenerated train_y (robust: y encodes noise+params)
    test_id = None
    obs_y = [p[1] for p in train_pairs]
    for tid in range(1, 200):
        gy = make_train_y(tid)
        if len(gy) == len(obs_y) and all(abs(g - o) <= 1e-6 * (1 + abs(g))
                                         for g, o in zip(gy, obs_y)):
            test_id = tid
            break
    if test_id is None:
        fail("unrecognized instance")

    # ---- read participant expression ----
    try:
        blob = open(out_path).read()
    except Exception:
        fail("no output")
    lines = [ln.strip() for ln in blob.splitlines() if ln.strip()]
    if not lines:
        fail("empty output")
    expr = lines[-1]
    if expr.count("=") == 1 and "==" not in expr:
        expr = expr.split("=", 1)[1].strip()   # tolerate "y = ..."

    try:
        ev, complexity = compile_expr(expr)
    except Exception as e:
        fail("unparseable/disallowed: %s" % e)

    if complexity > 400:
        fail("expression too complex")

    # ---- evaluate on held-out extrapolation region ----
    hx, hy = make_held(test_id)
    preds = []
    for x in hx:
        try:
            v = ev(x)
        except Exception as e:
            fail("eval error: %s" % e)
        v = float(v)
        if not math.isfinite(v):
            fail("non-finite prediction")
        if abs(v) > 1e6:
            fail("absurd prediction magnitude")
        preds.append(v)

    # participant extrapolation error with a mild complexity penalty
    err = rmse(preds, hy)
    eff = err * (1.0 + 0.004 * complexity)

    # internal trivial baseline B: predict the constant train-mean everywhere
    ty = make_train_y(test_id)
    tmean = sum(ty) / len(ty)
    B = rmse([tmean] * len(hy), hy)

    sc = min(1000.0, 100.0 * B / max(1e-9, eff))
    print("test_id=%d B=%.6g err=%.6g complexity=%d eff=%.6g" %
          (test_id, B, err, complexity, eff))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
