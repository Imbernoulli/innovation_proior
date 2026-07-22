import sys, ast, math

# ---------------------------------------------------------------------------
# fsx_B_0937 checker.
#
# The participant's artifact (<out>) is ONE line: a closed-form Python-style
# expression over variables {a, b, lam} (plus constants pi, e and functions
# sqrt/log/exp/abs/min/max) that predicts the idealized mode-count N(a,b,lam)
# of a rectangular pan of side lengths a,b at spectral parameter lam.
#
# The checker evaluates that expression on a HELD-OUT / EXTRAPOLATION split
# -- highly elongated "baking sheet" pans (aspect ratio 10-30) at moderate
# lam -- that is regenerated HERE, deterministically, from the fixed hidden
# law below. The train sample the solver saw (from gen.py) never contains
# this law, these coefficients, or these rows.
# ---------------------------------------------------------------------------

# ---- hidden law (lives ONLY here) -----------------------------------------
_PI = 3.141592653589793
C1 = 1.0 / (4.0 * _PI)   # bulk / area-scaling contribution
C2 = 1.0 / (4.0 * _PI)   # boundary / perimeter-correction contribution
C3 = 1.5                 # constant (corner-layer) contribution


def _n_true(a, b, lam):
    A = a * b
    P = 2.0 * (a + b)
    return C1 * A * lam - C2 * P * (lam ** 0.5) + C3


HELD_AREA = [5.0, 10.0, 18.0, 28.0, 40.0]
HELD_R    = [10.0, 15.0, 20.0, 25.0, 30.0]
HELD_LAM  = [30.0, 50.0, 80.0, 120.0]

GUESS_C = 0.092   # checker's own naive baseline coefficient (no fitting at all)
F_FLOOR = 0.05    # irreducible-uncertainty floor added to every submission's error
                  # (including the internal baseline), so no single test case can
                  # let a lucky fit saturate the score to 1.0


def held_out_rows():
    rows = []
    for Area in HELD_AREA:
        for r in HELD_R:
            a = (Area * r) ** 0.5
            b = (Area / r) ** 0.5
            for lam in HELD_LAM:
                rows.append((a, b, lam, _n_true(a, b, lam)))
    return rows


# ---- restricted expression evaluator ---------------------------------------
_ALLOWED_NAMES = {"a", "b", "lam", "pi", "e"}
_ALLOWED_FUNCS = {"sqrt", "log", "exp", "abs", "min", "max"}
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


def _check_ast(node):
    if isinstance(node, ast.Expression):
        _check_ast(node.body)
    elif isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINOPS):
            raise ValueError("op not allowed")
        _check_ast(node.left)
        _check_ast(node.right)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARYOPS):
            raise ValueError("unary op not allowed")
        _check_ast(node.operand)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise ValueError("call not allowed")
        if node.keywords:
            raise ValueError("kwargs not allowed")
        for arg in node.args:
            _check_ast(arg)
    elif isinstance(node, ast.Name):
        if node.id not in _ALLOWED_NAMES:
            raise ValueError("name not allowed: %s" % node.id)
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError("constant type not allowed")
    else:
        raise ValueError("node type not allowed: %s" % type(node).__name__)


def _count_nodes(node):
    return sum(1 for _ in ast.walk(node))


def compile_expr(text):
    text = text.strip()
    if not text or len(text) > 2000:
        raise ValueError("bad length")
    tree = ast.parse(text, mode="eval")
    _check_ast(tree)
    n_nodes = _count_nodes(tree)
    code = compile(tree, "<expr>", "eval")
    return code, n_nodes


def eval_expr(code, a, b, lam):
    ns = {"a": a, "b": b, "lam": lam, "pi": _PI, "e": math.e,
          "sqrt": math.sqrt, "log": math.log, "exp": math.exp,
          "abs": abs, "min": min, "max": max}
    val = eval(code, {"__builtins__": {}}, ns)
    val = float(val)
    if not math.isfinite(val):
        raise ValueError("non-finite")
    return val


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def nrmse(rows, code):
    se = 0.0
    ssq = 0.0
    for a, b, lam, nt in rows:
        pred = eval_expr(code, a, b, lam)
        se += (pred - nt) ** 2
        ssq += nt ** 2
    scale = math.sqrt(ssq / len(rows))
    if scale <= 1e-9:
        scale = 1e-9
    return math.sqrt(se / len(rows)) / scale


def main():
    if len(sys.argv) < 3:
        fail("bad args")
    out_path = sys.argv[2]

    try:
        with open(out_path, "r", errors="replace") as f:
            raw = f.read()
    except Exception:
        fail("cannot read output")

    if len(raw) > 4000:
        fail("output too long")

    lines = [ln for ln in raw.splitlines() if ln.strip() != ""]

    if not lines:
        fail("empty output")
    if len(lines) != 1:
        fail("output must be exactly one non-empty line, got %d" % len(lines))

    expr_text = lines[0]

    try:
        code, n_nodes = compile_expr(expr_text)
    except Exception as ex:
        fail("parse error: %s" % ex)

    rows = held_out_rows()

    try:
        F_err = nrmse(rows, code)
    except Exception as ex:
        fail("eval error: %s" % ex)

    NODE_BUDGET = 40
    penalty = max(0, n_nodes - NODE_BUDGET) * 0.003
    F = F_err + penalty + F_FLOOR

    # ---- internal baseline B: naive fixed-coefficient guess, no fitting ----
    def _guess(a, b, lam):
        return GUESS_C * a * b * lam
    se = 0.0
    ssq = 0.0
    for a, b, lam, nt in rows:
        pred = _guess(a, b, lam)
        se += (pred - nt) ** 2
        ssq += nt ** 2
    scale = math.sqrt(ssq / len(rows))
    if scale <= 1e-9:
        scale = 1e-9
    B = math.sqrt(se / len(rows)) / scale + F_FLOOR
    B = max(B, 1e-9)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    print("F=%.6f B=%.6f nodes=%d  Ratio: %.6f" % (F, B, n_nodes, ratio))
    sys.exit(0)


if __name__ == "__main__":
    main()
