#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  -- deterministic grader for the battery-knee-forecast
symbolic-regression problem (format E).

Reads the TRAIN log from <in> (only to build the constant-mean baseline), parses
the participant's expression from <out> under a strict whitelist, regenerates a
HELD-OUT set deterministically from the private ground truth -- cells cycled under
HARSHER conditions than anything in training, forecast out to a HORIZON of cycle
counts that pushes many of them well past their own (never-observed-in-training)
knee -- and scores held-out RMSE (complexity-inflated) against the constant-mean
baseline as an accuracy ratio.

Prints exactly one line ending in "Ratio: <r>" with r in [0,1]. Any infeasibility
(parse error, disallowed token, non-finite value anywhere) => Ratio: 0.0.
"""
import sys, math, random, ast

# ---------------- private ground truth (identical to gen.py's) ----------------
N0 = 900.0
KAPPA = 110.0
ALPHA = 0.00035
BETA = 0.006

R0 = 0.05
ETA = 0.55

HELD_TEMP_LO, HELD_TEMP_HI = 0.6, 1.0
HELD_DOD_LO, HELD_DOD_HI = 0.6, 1.0
HELD_CYC_LO, HELD_CYC_HI = 500.0, 950.0   # forward horizon: crosses the knee for most cells

HELDOUT_SEED = 7777
HELDOUT_N = 220
HELDOUT_R_NOISE = 0.12
HELDOUT_CAP_NOISE = 0.00008
HELDOUT_Y_NOISE = 0.05

MU = 0.0012          # complexity weight (mild parsimony tiebreaker)

ALLOWED_FUNCS = {
    "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "tanh": math.tanh,
    "abs": abs, "pow": pow,
}
ALLOWED_VARS = {"x0", "x1", "x2", "x3", "x4"}


def stress(temp, dod):
    return 1.0 + 1.3 * temp + 1.3 * dod + 1.6 * temp * dod


def n_knee(temp, dod):
    return N0 - KAPPA * stress(temp, dod)


def capacity_at(temp, dod, cyc):
    nk = n_knee(temp, dod)
    if cyc <= nk:
        return 1.0 - ALPHA * cyc
    y_knee = 1.0 - ALPHA * nk
    return y_knee * math.exp(-BETA * (cyc - nk))


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
    if len(expr) > 6000:
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
    if len(vals) < 6 or len(vals) % 6 != 0:
        bail("malformed train data")
    ys = [vals[i + 5] for i in range(0, len(vals), 6)]
    return ys


def gen_heldout():
    rng = random.Random(HELDOUT_SEED)
    pts = []
    for _ in range(HELDOUT_N):
        temp = rng.uniform(HELD_TEMP_LO, HELD_TEMP_HI)
        dod = rng.uniform(HELD_DOD_LO, HELD_DOD_HI)
        s = stress(temp, dod)
        x2 = R0 + ETA * s + rng.gauss(0.0, HELDOUT_R_NOISE)
        x3 = ALPHA + rng.gauss(0.0, HELDOUT_CAP_NOISE)
        cyc = rng.uniform(HELD_CYC_LO, HELD_CYC_HI)
        y = capacity_at(temp, dod, cyc) + rng.gauss(0.0, HELDOUT_Y_NOISE)
        pts.append((temp, dod, x2, x3, cyc, y))
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
    for (x0, x1, x2, x3, x4, y) in held:
        se_base += (mean_y - y) ** 2
        loc = {"x0": x0, "x1": x1, "x2": x2, "x3": x3, "x4": x4}
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

    E_base = rmse_base * (1.0 + MU * 1)
    E_part = rmse_part * (1.0 + MU * ncount)

    # accuracy ratio: F = -log(complexity-adjusted RMSE), clamped at 0, so an order-
    # of-magnitude RMSE reduction (typical for "beats the baseline a bit") registers
    # smoothly instead of the near-flat response an inverse-RMSE ratio gives once the
    # baseline error is already large (as it is here: the held-out set spans both
    # still-linear and fully-collapsed cells). The constant-mean predictor (same
    # complexity convention, C=1) exactly anchors the ratio at ~0.1 by construction
    # (F_part == F_base when the submission reproduces that baseline exactly).
    F_base = max(0.0, -math.log(max(1e-9, E_base)))
    F_part = max(0.0, -math.log(max(1e-9, E_part)))
    sc = min(1000.0, 100.0 * F_part / max(1e-9, F_base))
    ratio = sc / 1000.0
    print("rmse=%.6f C=%d F_base=%.6f F_part=%.6f Ratio: %.6f" % (rmse_part, ncount, F_base, F_part, ratio))
    sys.exit(0)


if __name__ == "__main__":
    main()
