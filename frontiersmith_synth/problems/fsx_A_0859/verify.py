#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the jeweler's-wobbling-scale variance-law task.

- Reads (n_x, R, mu, t) and the training log from <in>.
- Regenerates the hidden two-regime wobble law (x0, p, A, C) and a fresh set of
  HELD-OUT, HEAVIER-load readings entirely from t (never printed anywhere; the
  ground truth lives ONLY here and is regenerated with a seed stream disjoint
  from gen.py's training-data seed stream).
- Parses the participant's closed-form expression g(x) for the predicted
  variance, in a restricted grammar: + - * / ** unary minus, parentheses,
  numeric constants, the variable x, and the unary functions
  abs, sqrt, exp, log, step (Heaviside: 1 if arg>0 else 0).
- Feasibility: must parse, must use only allowed names/constants, must be
  <= MAX_NODES nodes, and must evaluate to a FINITE, STRICTLY POSITIVE value
  at every held-out load. Any violation -> Ratio: 0.0.
- Score (minimise): mean Gaussian negative log-likelihood of the true held-out
  readings under N(mu, g(x)):
      F = mean_i [ 0.5*log(2*pi*g(x_i)) + (y_i-mu)^2 / (2*g(x_i)) ]
  Baseline B = the same NLL formula evaluated with a CONSTANT variance equal to
  the pooled empirical variance of the TRAINING residuals about mu (the
  "wobble doesn't depend on load" baseline, built by the checker itself from
  <in>). Ratio = min(1000, 100*B/F) / 1000.
"""
import sys, math, ast, random

MU_HDR_TOL = 1e-6
XLO, XHI = 0.3, 5.0
MAX_NODES = 40
MAX_OUT_BYTES = 20000
NH_X = 24          # distinct held-out loads
RH = 16            # repeats per held-out load


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden wobble law (identical to gen.py) ----------
def hidden_params(t):
    rng = random.Random(3010301 + t * 92821)
    x0 = rng.uniform(2.6, 3.9)
    p = rng.uniform(0.05, 0.35)
    A = rng.uniform(0.5, 1.2)
    C = rng.uniform(1.0, 2.2)
    return x0, p, A, C


def wobble(x, x0, p, A, C):
    if x < x0:
        return A * (x ** p)
    return A * (x0 ** p) + C * (x - x0)


def heldout_data(t, mu):
    """Held-out HEAVY-load readings; disjoint seed stream from gen.py; never printed."""
    x0, p, A, C = hidden_params(t)
    ext = 9.5 + 1.6 * (t - 1)           # heavier extrapolation reach as t grows
    lo, hi = XHI, XHI + ext
    rng = random.Random(88014721 + t * 4013)
    xs, ys = [], []
    for _ in range(NH_X):
        x = rng.uniform(lo, hi)
        var = wobble(x, x0, p, A, C)
        sd = math.sqrt(max(1e-12, var))
        for _ in range(RH):
            xs.append(x)
            ys.append(mu + rng.gauss(0.0, sd))
    return xs, ys


# ---------- expression grammar ----------
ALLOWED_FUNCS = {
    "abs": abs,
    "sqrt": lambda u: math.sqrt(u) if u >= 0 else float("nan"),
    "exp": lambda u: math.exp(u) if u < 700 else float("inf"),
    "log": lambda u: math.log(u) if u > 0 else float("nan"),
    "step": lambda u: 1.0 if u > 0 else 0.0,
}
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
)


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
            if node.id not in ("x",) and node.id not in ALLOWED_FUNCS:
                return "unknown name %s" % node.id
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                return "non-numeric constant"
            v = float(node.value)
            if v != v or v in (float("inf"), float("-inf")):
                return "non-finite constant"
    return None


def _count_nodes(tree):
    return sum(1 for nd in ast.walk(tree)
               if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant)))


def compile_expr(text):
    text = text.strip()
    if not text:
        fail("empty expression")
    if len(text) > 2000:
        fail("expression text too long")
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


def eval_g(code, xv):
    env = dict(ALLOWED_FUNCS)
    env["x"] = xv
    try:
        with __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore")
            val = eval(code, {"__builtins__": {}}, env)
    except Exception:
        fail("evaluation error at x=%.6f" % xv)
    if not isinstance(val, (int, float)) or isinstance(val, bool):
        fail("non-numeric result at x=%.6f" % xv)
    val = float(val)
    if val != val or val in (float("inf"), float("-inf")):
        fail("non-finite variance at x=%.6f" % xv)
    if val <= 0.0:
        fail("non-positive variance at x=%.6f" % xv)
    return val


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            lines = fh.read().splitlines()
        hdr = lines[0].split()
        n_x, R, mu, t = int(hdr[0]), int(hdr[1]), float(hdr[2]), int(hdr[3])
        train_y = []
        for ln in lines[1:1 + n_x]:
            parts = ln.split()
            for v in parts[1:1 + R]:
                train_y.append(float(v))
    except Exception:
        fail("bad instance file")
    if t < 1 or t > 100000 or n_x <= 0 or R <= 0 or not train_y:
        fail("bad instance parameters")

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    raw_text = raw.decode("utf-8", "replace")
    # statement says "exactly one line": strip AT MOST one trailing line
    # terminator first (do NOT blanket .strip(), which would also eat
    # leading/embedded blank lines and mask a multi-line submission), THEN
    # require no newline/carriage-return survives anywhere in what's left.
    if raw_text.endswith("\r\n"):
        raw_text = raw_text[:-2]
    elif raw_text.endswith("\n") or raw_text.endswith("\r"):
        raw_text = raw_text[:-1]
    if "\n" in raw_text or "\r" in raw_text:
        fail("output must be exactly one line")
    text = raw_text.strip(" \t")
    if not text:
        fail("empty output")
    code, nodes = compile_expr(text)

    xs_h, ys_h = heldout_data(t, mu)

    # baseline: pooled empirical variance of TRAINING residuals about mu
    base_var = sum((yv - mu) ** 2 for yv in train_y) / len(train_y)
    base_var = max(1e-9, base_var)

    def nll_mean(xs, ys, g_of_x):
        total = 0.0
        for xv, yv in zip(xs, ys):
            gv = g_of_x(xv)
            total += 0.5 * math.log(2.0 * math.pi * gv) + (yv - mu) ** 2 / (2.0 * gv)
        return total / len(ys)

    F = nll_mean(xs_h, ys_h, lambda xv: eval_g(code, xv))
    B = nll_mean(xs_h, ys_h, lambda xv: base_var)

    if F != F or F in (float("inf"), float("-inf")):
        fail("non-finite score")

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("heldout_NLL=%.6f baseline_NLL=%.6f nodes=%d  Ratio: %.6f"
          % (F, B, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
