#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  -- deterministic grader for the ISP congestion
symbolic-regression problem (format E).

Reads the LOW-LOAD training log from <in> (only to build the constant-mean
baseline), parses the participant's expression from <out> under a strict
whitelist, regenerates the HELD-OUT HIGH-LOAD extrapolation split
deterministically from the private ground truth, and scores held-out RMSE
(complexity-inflated) against the constant-mean baseline.

Prints exactly one line ending in "Ratio: <r>" with r in [0,1]. Any infeasibility
(parse error, disallowed token, non-finite value anywhere) => Ratio: 0.0.
"""
import sys, math, random, ast

# ---------------- private ground truth (identical to gen.py's) ----------------
def hidden_law(rho, cv, hop):
    q = rho / (1.0 - rho)
    return 0.5 + 0.9 * hop + 1.3 * (1.0 + 0.6 * cv) * q

# held-out HIGH-LOAD extrapolation regime: utilization pushed into [0.66, 0.86],
# a range NEVER seen in training (train rho <= 0.60). cv/hop share the training
# range (those channels ARE observed across the full operating envelope). Fixed
# seed + fixed irreducible noise -> fully deterministic.
HELDOUT_SEED = 20250703
HELDOUT_N = 240
RHO_LO, RHO_HI = 0.66, 0.86
CV_LO, CV_HI = 0.0, 1.5
HOP_LO, HOP_HI = 0.0, 1.0
HELDOUT_NOISE = 1.6          # irreducible high-load measurement noise
MU = 0.002                   # complexity weight (mild parsimony tiebreaker)

ALLOWED_FUNCS = {
    "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "tanh": math.tanh,
    "abs": abs, "pow": pow,
}
ALLOWED_VARS = {"rho", "cv", "hop"}


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
        if nd.__class__.__name__ == "Num":   # py<3.8 literal compat
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
    if len(vals) < 4 or len(vals) % 4 != 0:
        bail("malformed train data")
    ys = [vals[i + 3] for i in range(0, len(vals), 4)]
    return ys


def gen_heldout():
    rng = random.Random(HELDOUT_SEED)
    pts = []
    for _ in range(HELDOUT_N):
        rho = rng.uniform(RHO_LO, RHO_HI)
        cv = rng.uniform(CV_LO, CV_HI)
        hop = rng.uniform(HOP_LO, HOP_HI)
        y = hidden_law(rho, cv, hop) + rng.gauss(0.0, HELDOUT_NOISE)
        pts.append((rho, cv, hop, y))
    return pts


def main():
    if len(sys.argv) < 3:
        bail("usage")
    train_ys = read_train(sys.argv[1])
    code, ncount = read_expr(sys.argv[2])

    held = gen_heldout()

    mean_y = sum(train_ys) / len(train_ys)
    se_base = 0.0
    se_part = 0.0
    g = {"__builtins__": {}}
    g.update(ALLOWED_FUNCS)
    for (rho, cv, hop, y) in held:
        se_base += (mean_y - y) ** 2
        loc = {"rho": rho, "cv": cv, "hop": hop}
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
