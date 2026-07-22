#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for "plague scrolls counted only on market days".

Regenerates the hidden latent contagion (pre-decree growth rate `beta`,
initial load `X0`), the fixed reporting operator (weekly market-day rhythm
`w[0..6]`, scribe-capacity saturation `Cap`) and the post-decree growth
multiplier `factor`, all deterministically from the test id embedded in the
instance header (identical formulas to gen.py -- duplicated on purpose, not
imported, per the no-groundtruth-module rule).

The participant's stdout is ONE Python arithmetic expression over the
variables `t` (day index), `n` (number of training days = the day the decree
takes effect) and `f` (the known post-decree growth multiplier), using only
`+ - * / %`, parentheses, numeric constants, a bracketed 7-element list with a
`[...][expr]` subscript (for a day-of-week lookup), and the unary functions
`exp`, `log`, `sqrt`, `absv`. It is rolled out on HELD-OUT days
t = n .. n+H-1 (a genuine extrapolation region into the post-decree regime,
never shown to the solver) and scored by held-out RMSE with a light
node-count parsimony penalty (minimisation):

    F = heldout_RMSE * (1 + LAMBDA * nodes)
    B = baseline_RMSE * (1 + LAMBDA * 1)     # baseline: flat last-training-week mean
    Ratio = min(1000, 100*B/F) / 1000

A flat continuation reproduces the baseline (~0.1). Fitting the reported
counts as if they directly were the epidemic state absorbs the market
rhythm and the scribe saturation into a biased "trend" that the announced
factor cannot correctly rescale; a model that first separates the growth
rate from the (decree-invariant) reporting operator, then rescales ONLY the
growth rate by `f`, extrapolates correctly.
"""
import sys, math, ast, random

LAMBDA = 0.002
MAX_NODES = 140
MAX_OUT_BYTES = 20000

ALLOWED_FUNCS = {
    "exp": lambda x: math.exp(max(-700.0, min(700.0, x))),
    "log": math.log,
    "sqrt": math.sqrt,
    "absv": abs,
}
ALLOWED_NAMES = {"t", "n", "f"}

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod,
    ast.USub, ast.UAdd, ast.List,
    ast.Subscript,
)


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden epidemic + reporting operator (identical to gen.py) ----------
def get_params(tid):
    rng = random.Random(900001 + tid * 104729)
    diff = (tid - 1) / 9.0

    beta = 0.070 + 0.035 * rng.random()
    X0 = 4.0 + 6.0 * rng.random()
    factor = 0.30 + 0.30 * rng.random()
    n = 28 + 2 * (tid % 6)
    H = 12 + (tid % 4)

    base_w = [1.15, 0.95, 0.80, 0.90, 1.05, 1.35, 0.55]
    amp = 0.35 + 0.55 * diff
    w = []
    for k in range(7):
        jitter = 1.0 + (rng.random() - 0.5) * 0.25
        val = base_w[k] * jitter
        if k == 6:
            val *= (1.0 - 0.55 * amp)
        w.append(max(0.04, val))
    m = sum(w) / 7.0
    w = [v / m for v in w]

    peak_load = X0 * math.exp(beta * (n - 1)) * max(w)
    cap_ratio = max(1.08, 4.2 - 3.0 * diff)
    Cap = peak_load * cap_ratio

    return dict(beta=beta, X0=X0, factor=factor, n=n, H=H, w=w, Cap=Cap)


def true_load(t, p):
    n, beta, X0, factor, w = p["n"], p["beta"], p["X0"], p["factor"], p["w"]
    if t < n:
        X = X0 * math.exp(beta * t)
    else:
        X_n = X0 * math.exp(beta * n)
        X = X_n * math.exp(beta * factor * (t - n))
    dow = t % 7
    return X * w[dow]


def true_reported(t, p, tid, noise=True):
    load = true_load(t, p)
    Cap = p["Cap"]
    base = Cap * load / (Cap + load)
    if noise:
        nrng = random.Random(31337 * tid + 7 * t + 991)
        eps = (nrng.random() - 0.5) * 0.18
        base *= (1.0 + eps)
    return max(0.0, base)


def rounded_reported(t, p, tid, noise=True):
    return float(int(round(true_reported(t, p, tid, noise=noise))))


# ---------- expression parsing / validation ----------
def _validate_ast(tree):
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return "disallowed syntax %s" % type(node).__name__
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS):
                return "disallowed call"
            if node.keywords or len(node.args) != 1:
                return "bad function arity"
        if isinstance(node, ast.Name):
            nm = node.id
            if nm in ALLOWED_FUNCS:
                continue
            if nm not in ALLOWED_NAMES:
                return "unknown name %s" % nm
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                return "non-numeric constant"
            v = float(node.value)
            if v != v or v in (float("inf"), float("-inf")):
                return "non-finite constant"
        if isinstance(node, ast.List):
            # A list literal may ONLY be the stated 7-slot day-of-week table --
            # this blocks a length-H "answer key" lookup keyed directly off a
            # held-out day offset (which would bypass the intended inference).
            if len(node.elts) != 7:
                return "list literal must have exactly 7 elements (day-of-week table)"
    return None


def _count_nodes(tree):
    return sum(1 for nd in ast.walk(tree)
               if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name,
                                   ast.Constant, ast.List, ast.Subscript)))


def compile_expr(text):
    text = text.strip()
    if not text:
        fail("empty expression")
    if len(text.splitlines()) != 1:
        fail("expression must be a single line")
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    err = _validate_ast(tree)
    if err:
        fail(err)
    nodes = _count_nodes(tree)
    if nodes > MAX_NODES:
        fail("expression too large (%d nodes)" % nodes)
    try:
        code = compile(tree, "<expr>", "eval")
    except Exception:
        fail("compile error")
    return code, nodes


def eval_expr(code, tval, nval, fval):
    # t and n stay INT (so `t % 7` can index a list literal); f stays float.
    env = dict(ALLOWED_FUNCS)
    env["t"] = int(tval)
    env["n"] = int(nval)
    env["f"] = float(fval)
    try:
        v = eval(code, {"__builtins__": {}}, env)
    except Exception:
        return None
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return None
    v = float(v)
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v


def rmse(pred, true):
    return math.sqrt(sum((p - y) ** 2 for p, y in zip(pred, true)) / len(true))


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            header = fh.readline().split()
        n_hdr = int(header[0]); tid = int(header[1]); f_hdr = float(header[2])
    except Exception:
        fail("bad instance header")
    if tid < 1 or tid > 1000000:
        fail("bad test id")

    p = get_params(tid)
    if p["n"] != n_hdr or abs(p["factor"] - f_hdr) > 1e-4:
        fail("instance/header mismatch")
    n, factor, H = p["n"], p["factor"], p["H"]

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace")

    code, nodes = compile_expr(text)

    heldout_t = list(range(n, n + H))
    true_vals = [rounded_reported(t, p, tid, noise=True) for t in heldout_t]
    pred_vals = []
    for t in heldout_t:
        v = eval_expr(code, float(t), float(n), float(factor))
        if v is None:
            fail("non-finite or invalid evaluation at t=%d" % t)
        pred_vals.append(v)

    base_vals = [rounded_reported(t, p, tid, noise=True) for t in range(n - 7, n)]
    baseline_pred = sum(base_vals) / len(base_vals)

    F_err = rmse(pred_vals, true_vals)
    B_err = rmse([baseline_pred] * len(true_vals), true_vals)

    F = F_err * (1.0 + LAMBDA * nodes)
    B = B_err * (1.0 + LAMBDA * 1)
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("heldout_RMSE=%.6f baseline_RMSE=%.6f nodes=%d  Ratio: %.6f"
          % (F_err, B_err, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
