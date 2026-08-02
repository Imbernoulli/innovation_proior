#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the churn-hazard-forecast task.

- Reads N, T_obs, testId from the <in> header (the visible censored sample
  itself is not needed here: the hidden law is looked up analytically from
  testId, identically to gen.py -- the ground truth lives ONLY in this file
  and in gen.py, never in an importable module).
- Parses the participant's OUTPUT: a single closed-form expression string
  over variables t (tenure) and x (cohort covariate), the constants pi/e,
  arithmetic + - * / ** and unary -, and the functions exp/log/sqrt/abs/min/max.
  Validated with a strict AST whitelist (no attribute access, no comprehensions,
  no arbitrary calls) -- unknown syntax, unknown names, wrong arities, or any
  non-finite / non-real value produced anywhere on the held-out grid -> Ratio 0.
- Regenerates a HELD-OUT grid of (t, x) points that genuinely EXTRAPOLATES
  beyond the visible window (t up to 3x T_obs) and spans the full cohort range
  (x on a finer grid than the six training buckets), evaluates the true
  survival probability S(t,x) = exp(-(t/lambda(x))**kappa(x)) analytically,
  and scores the mean absolute error of the submitted expression against it
  (plus a light parsimony tax), relative to the checker's own constant-0.5
  baseline (minimisation form):
      F = MAE + penalty(nodes)
      B = MAE_of_constant_0.5
      Ratio = min(1000, 100*B/F) / 1000
  A flat 0.5 guess reproduces the baseline (~0.1).  A model that treats the
  observed-tenure sample as if uncensored, or ignores the cohort covariate,
  is systematically biased and stays low, especially once t runs past T_obs
  where nothing but the correctly-recovered hazard SHAPE can extrapolate.
"""
import sys, math, ast, random

MAX_OUT_BYTES = 20000
MAX_NODES = 80
PENALTY_FREE_NODES = 60
PENALTY_PER_NODE = 0.002

# Must be identical to gen.py's PARAMS (the hidden law + instance sizing).
PARAMS = {
    1:  (400,  40, 1.40,  0.40, 25.0,  0.10, (0.20, 0.20, 0.20, 0.16, 0.14, 0.10)),
    2:  (500,  25, 0.55,  0.15, 30.0,  0.20, (0.30, 0.25, 0.20, 0.15, 0.07, 0.03)),
    3:  (350,  60, 1.10, -0.50, 20.0, -0.15, (0.15, 0.15, 0.15, 0.15, 0.20, 0.20)),
    4:  (800,  20, 0.50,  0.00, 45.0,  0.00, (0.166667,) * 6),
    5:  (300,  22, 1.80, -0.90, 15.0,  0.30, (0.10, 0.10, 0.15, 0.20, 0.20, 0.25)),
    6:  (600,  35, 0.60,  0.60, 22.0, -0.25, (0.05, 0.10, 0.15, 0.20, 0.25, 0.25)),
    7:  (250,  18, 0.45,  0.05, 50.0,  0.05, (0.166667,) * 6),
    8:  (1200, 28, 1.60, -1.00, 18.0,  0.35, (0.30, 0.05, 0.05, 0.05, 0.05, 0.50)),
    9:  (700,  30, 0.70,  0.90, 28.0, -0.40, (0.35, 0.25, 0.15, 0.10, 0.10, 0.05)),
    10: (1500, 22, 0.40,  1.40, 35.0,  0.15, (0.50, 0.02, 0.02, 0.02, 0.02, 0.42)),
}
T_FRACS = (0.4, 0.7, 1.0, 1.5, 2.0, 3.0)   # includes genuine extrapolation past T_obs
X_GRID = tuple(i / 10.0 for i in range(11))  # finer than the 6 training buckets

ALLOWED_FUNCS = {
    "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
    "abs": abs, "min": min, "max": max,
}
ALLOWED_CONSTS = {"pi": math.pi, "e": math.e}
ALLOWED_VARS = {"t", "x"}
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
)


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


NOISE_SIGMA = 0.05  # fixed irreducible measurement-noise floor on the held-out label


def true_survival(t, x, kappa0, kappa1, lam0, lam1):
    kappa = kappa0 + kappa1 * x
    lam = lam0 * math.exp(lam1 * x)
    return math.exp(-((t / lam) ** kappa))


def held_out_grid(T_obs):
    return [(f * T_obs, x) for f in T_FRACS for x in X_GRID]


def noisy_label(t_id, idx, s_true):
    """Deterministic, submission-independent measurement noise on the grading
    label itself -- an irreducible floor so even a perfectly-recovered hazard
    shape cannot drive held-out error to exactly 0 (keeps scoring headroom)."""
    rng = random.Random(9130007 + t_id * 104729 + idx * 65537)
    v = s_true + rng.gauss(0.0, NOISE_SIGMA)
    return max(0.0, min(1.0, v))


def validate_ast(tree):
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            fail("disallowed syntax %s" % type(node).__name__)
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS):
                fail("disallowed call")
            if node.keywords:
                fail("keyword args not allowed")
            fname = node.func.id
            nargs = len(node.args)
            if fname in ("exp", "sqrt", "abs"):
                if nargs != 1:
                    fail("bad arity for %s" % fname)
            elif fname == "log":
                if nargs not in (1, 2):
                    fail("bad arity for log")
            elif fname in ("min", "max"):
                if nargs < 2:
                    fail("bad arity for %s" % fname)
        if isinstance(node, ast.Name):
            nm = node.id
            if nm in ALLOWED_FUNCS or nm in ALLOWED_CONSTS or nm in ALLOWED_VARS:
                pass
            else:
                fail("unknown name %s" % nm)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                fail("non-numeric constant")
            v = float(node.value)
            if v != v or v in (float("inf"), float("-inf")):
                fail("non-finite constant")


def count_nodes(tree):
    return sum(1 for nd in ast.walk(tree)
               if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant)))


def compile_expr(text):
    text = text.strip()
    if not text:
        fail("empty expression")
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    validate_ast(tree)
    nodes = count_nodes(tree)
    if nodes > MAX_NODES:
        fail("expression too large (%d nodes)" % nodes)
    try:
        code = compile(tree, "<expr>", "eval")
    except Exception:
        fail("compile error")
    return code, nodes


def eval_at(code, t, x):
    env = dict(ALLOWED_FUNCS)
    env.update(ALLOWED_CONSTS)
    env["t"] = t
    env["x"] = x
    try:
        v = eval(code, {"__builtins__": {}}, env)
    except Exception:
        fail("evaluation error at t=%.4f x=%.2f" % (t, x))
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        fail("non-real result at t=%.4f x=%.2f" % (t, x))
    v = float(v)
    if v != v or v in (float("inf"), float("-inf")):
        fail("non-finite result at t=%.4f x=%.2f" % (t, x))
    return v


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            header = fh.readline().split()
        T_obs = float(header[1])
        t_id = int(header[2])
    except Exception:
        fail("bad instance header")
    if t_id not in PARAMS:
        fail("bad test id")
    _, T_obs_ref, kappa0, kappa1, lam0, lam1, _ = PARAMS[t_id]
    T_obs = float(T_obs_ref)  # canonical, ignore whatever the input claims

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace")
    first_line = next((ln for ln in text.splitlines() if ln.strip()), "")

    code, nodes = compile_expr(first_line)

    grid = held_out_grid(T_obs)
    abs_err = 0.0
    base_err = 0.0
    for idx, (tt, xx) in enumerate(grid):
        s_true = true_survival(tt, xx, kappa0, kappa1, lam0, lam1)
        s_label = noisy_label(t_id, idx, s_true)
        s_hat = eval_at(code, tt, xx)
        s_hat_clipped = max(0.0, min(1.0, s_hat))
        abs_err += abs(s_hat_clipped - s_label)
        base_err += abs(0.5 - s_label)
    m = len(grid)
    mae = abs_err / m
    base_mae = base_err / m

    penalty = PENALTY_PER_NODE * max(0, nodes - PENALTY_FREE_NODES)
    F = mae + penalty
    B = base_mae

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("mae=%.6f baseline_mae=%.6f nodes=%d  Ratio: %.6f"
          % (mae, base_mae, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
