#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  -- deterministic grader for the gravitational-lensing
radial-force-law recovery problem (format E).

Reads the INNER-band training profile from <in>, parses the participant's closed-form
expression from <out> under a strict whitelist, regenerates the HELD-OUT OUTER
(weak-lensing) EXTRAPOLATION band deterministically from the private ground truth, and
scores held-out RMSE (complexity-inflated) against an internal baseline predictor.

Baseline predictor B: the naive leading-only inverse-square fit c/r^2 (Newtonian point
mass, correction ignored), with c obtained by exact least squares on the training band.
This is a positive, feasible, beatable construction the grader builds itself.

Prints exactly one line ending in "Ratio: <r>" with r in [0,1].  Any infeasibility
(parse error, disallowed token, non-finite value anywhere) => Ratio: 0.0.
"""
import sys, math, random, ast

# ---------------- private ground truth (identical to gen.py's) ----------------
def hidden_law(r):
    A = 2.0
    C = 0.06
    return A / (r * r) + C / (r * r * r * r)

# Held-out OUTER weak-lensing band: r in [1.5, 4.0] (never sampled in training,
# which lives in [0.2, 0.6]); fixed seed + fixed irreducible noise -> deterministic.
HELDOUT_SEED = 4242
HELDOUT_N = 200
HELDOUT_LO = 1.5
HELDOUT_HI = 4.0
HELDOUT_NOISE = 0.05      # absolute irreducible measurement noise (headroom)
MU = 0.002                # complexity weight (mild parsimony tiebreaker)

ALLOWED_FUNCS = {
    "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "tanh": math.tanh,
    "abs": abs, "pow": pow,
}
ALLOWED_VARS = {"r"}


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
        if nd.__class__.__name__ == "Num":   # py<3.8 compat
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
    if len(vals) < 2 or len(vals) % 2 != 0:
        bail("malformed train data")
    rows = [(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]
    return rows


def gen_heldout():
    rng = random.Random(HELDOUT_SEED)
    pts = []
    for _ in range(HELDOUT_N):
        r = rng.uniform(HELDOUT_LO, HELDOUT_HI)
        y = hidden_law(r) + rng.gauss(0.0, HELDOUT_NOISE)
        pts.append((r, y))
    return pts


def leading_only_coeff(rows):
    # exact least squares for single basis phi = r^-2:  c = sum(F*phi)/sum(phi^2)
    num = 0.0
    den = 0.0
    for (r, f) in rows:
        phi = 1.0 / (r * r)
        num += f * phi
        den += phi * phi
    return num / den if den > 0 else 0.0


def main():
    if len(sys.argv) < 3:
        bail("usage")
    train = read_train(sys.argv[1])
    code, ncount = read_expr(sys.argv[2])

    held = gen_heldout()
    c_lead = leading_only_coeff(train)

    se_base = 0.0
    se_part = 0.0
    g = {"__builtins__": {}}
    g.update(ALLOWED_FUNCS)
    for (r, y) in held:
        base_pred = c_lead / (r * r)
        se_base += (base_pred - y) ** 2
        loc = {"r": r}
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

    # leading-only baseline expression "c/r**2" has an effective complexity ~4
    B = rmse_base * (1.0 + MU * 4)
    E = rmse_part * (1.0 + MU * ncount)

    sc = min(1000.0, 100.0 * B / max(1e-9, E))
    ratio = sc / 1000.0
    print("rmse=%.6f C=%d B=%.6f E=%.6f Ratio: %.6f" % (rmse_part, ncount, B, E, ratio))
    sys.exit(0)


if __name__ == "__main__":
    main()
