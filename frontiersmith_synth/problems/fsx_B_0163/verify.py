#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  -- deterministic grader for the polar-base
symbolic-regression problem (format E).

Reads the TRAIN log from <in> (only to build the constant-mean baseline), parses
the participant's expression from <out> under a strict whitelist, regenerates the
HELD-OUT EXTRAPOLATION split deterministically from the private ground truth, and
scores held-out RMSE (complexity-inflated) against the constant-mean baseline.

Prints exactly one line ending in "Ratio: <r>" with r in [0,1]. Any infeasibility
(parse error, disallowed token, non-finite value anywhere) => Ratio: 0.0.
"""
import sys, math, random, ast

# ---------------- private ground truth (identical to gen.py's) ----------------
def hidden_law(x0, x1, x2, x3):
    return 1.6 * x0 * x0 + 1.1 * x1 * x2 + 1.4 * math.exp(0.45 * x3) - 0.8 * x1 + 0.5

# held-out EXTRAPOLATION regime: every channel in [1.0, 1.6] (never seen in train,
# which lives in [-1,1]); fixed seed + fixed irreducible noise -> deterministic.
HELDOUT_SEED = 999
HELDOUT_N = 200
HELDOUT_LO = 1.0
HELDOUT_HI = 1.6
HELDOUT_NOISE = 0.6
MU = 0.002         # complexity weight (mild parsimony tiebreaker)

ALLOWED_FUNCS = {
    "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "tanh": math.tanh,
    "abs": abs, "pow": pow,
}
ALLOWED_VARS = {"x0", "x1", "x2", "x3"}


def bail(reason):
    print("infeasible: %s -- Ratio: 0.0" % reason)
    sys.exit(0)


# ---------------- strict expression whitelist ----------------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.USub, ast.UAdd, ast.Name,
)


def validate_ast(node):
    """Return (ok, ncount). Rejects anything outside the whitelist."""
    ncount = 0
    for nd in ast.walk(node):
        ncount += 1
        if isinstance(nd, ast.Constant):
            if not isinstance(nd.value, (int, float)) or isinstance(nd.value, bool):
                return False, 0
            continue
        # py<3.8 numeric literal compat (harmless on newer)
        if nd.__class__.__name__ == "Num":
            continue
        if isinstance(nd, ast.Name):
            if nd.id in ALLOWED_VARS or nd.id in ALLOWED_FUNCS:
                continue
            return False, 0
        if isinstance(nd, ast.Call):
            if not isinstance(nd.func, ast.Name) or nd.func.id not in ALLOWED_FUNCS:
                return False, 0
            if nd.keywords:
                return False, 0
            continue
        if not isinstance(nd, _ALLOWED_NODES):
            return False, 0
    return True, ncount


def read_expr(path):
    try:
        raw = open(path, "r", errors="replace").read()
    except Exception:
        bail("cannot read output")
    # take first non-empty line; strip an optional "y =" prefix
    expr = ""
    for line in raw.splitlines():
        s = line.strip()
        if s:
            expr = s
            break
    if not expr:
        bail("empty output")
    if "=" in expr:
        expr = expr.split("=", 1)[1].strip()
    if not expr:
        bail("empty expression")
    if len(expr) > 4000:
        bail("expression too long")
    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        bail("parse error")
    ok, ncount = validate_ast(tree)
    if not ok:
        bail("disallowed token in expression")
    try:
        code = compile(tree, "<expr>", "eval")
    except Exception:
        bail("compile error")
    return code, max(1, ncount)


def read_train(path):
    try:
        toks = open(path, "r", errors="replace").read().split()
    except Exception:
        bail("cannot read input")
    vals = []
    for tk in toks:
        try:
            vals.append(float(tk))
        except ValueError:
            bail("bad train token")
    if len(vals) < 5 or len(vals) % 5 != 0:
        bail("malformed train data")
    ys = [vals[i + 4] for i in range(0, len(vals), 5)]
    return ys


def gen_heldout():
    rng = random.Random(HELDOUT_SEED)
    pts = []
    for _ in range(HELDOUT_N):
        x0 = rng.uniform(HELDOUT_LO, HELDOUT_HI)
        x1 = rng.uniform(HELDOUT_LO, HELDOUT_HI)
        x2 = rng.uniform(HELDOUT_LO, HELDOUT_HI)
        x3 = rng.uniform(HELDOUT_LO, HELDOUT_HI)
        y = hidden_law(x0, x1, x2, x3) + rng.gauss(0.0, HELDOUT_NOISE)
        pts.append((x0, x1, x2, x3, y))
    return pts


def main():
    if len(sys.argv) < 3:
        bail("usage")
    train_ys = read_train(sys.argv[1])
    code, ncount = read_expr(sys.argv[2])

    held = gen_heldout()

    # baseline: constant train-mean predictor (C = 1)
    mean_y = sum(train_ys) / len(train_ys)
    se_base = 0.0
    se_part = 0.0
    g = {"__builtins__": {}}
    g.update(ALLOWED_FUNCS)
    for (x0, x1, x2, x3, y) in held:
        se_base += (mean_y - y) ** 2
        loc = {"x0": x0, "x1": x1, "x2": x2, "x3": x3}
        try:
            pred = eval(code, g, loc)
        except Exception:
            bail("evaluation error at held-out point")
        try:
            pred = float(pred)
        except (TypeError, ValueError):
            bail("non-numeric prediction")
        if not math.isfinite(pred):
            bail("non-finite prediction")
        se_part += (pred - y) ** 2

    rmse_base = math.sqrt(se_base / len(held))
    rmse_part = math.sqrt(se_part / len(held))

    B = rmse_base * (1.0 + MU * 1)
    E = rmse_part * (1.0 + MU * ncount)

    sc = min(1000.0, 100.0 * B / max(1e-9, E))
    ratio = sc / 1000.0
    print("rmse=%.6f C=%d B=%.6f E=%.6f Ratio: %.6f" % (rmse_part, ncount, B, E, ratio))
    sys.exit(0)


if __name__ == "__main__":
    main()
