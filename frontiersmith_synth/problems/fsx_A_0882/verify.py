#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the regime-switching scaling-law recovery task.

- Reads the test id from <in>, then regenerates the hidden regime-switching
  law
      x <  xc :  y = a0 + a1*x + a2*x^2
      x >= xc :  y = y(xc) + B*(x-xc)^alpha
  entirely from that id (IDENTICAL routine to gen.py -- lives ONLY here and
  in gen.py, never printed to the solver).
- Regenerates a HELD-OUT extrapolation grid: feed rates reaching FAR past the
  training band's excess into the congestion-cascade regime (offsets past xc
  of 40 .. ~1450, several decades beyond the training band's own excess of
  15), including a handful of extreme "adversarial corner" points. This grid
  is never shown to the solver.
- Parses the participant's closed-form output law -- a single expression over
  the variable `x`, numeric constants, + - * /, unary +/-, parentheses, and
  the functions absv(a) and powv(a,b) [a must evaluate to a positive
  finite number].
- Scores by mean SQUARED LOG ERROR between the law's prediction and the
  (noisy) held-out truth (rewards recovering the correct GROWTH RATE, i.e.
  the scaling exponent, not just matching one scale), with a small
  node-count parsimony penalty, against the flat geometric-mean-of-training
  baseline:
      F = mean_k (log(pred_k) - log(true_noisy_k))^2 * (1 + LAMBDA*nodes)
      Bs= mean_k (log(ybar_train) - log(true_noisy_k))^2 * (1 + LAMBDA*1)
      Ratio = min(SCORE_CAP, 0.1 * (Bs/F) ** GAMMA)
  Reproducing the flat training average everywhere gives Bs/F == 1, i.e.
  Ratio == 0.1. Held-out noise and the finite training sample keep even the
  correct law below the hard cap -- there is room above the reference
  solutions. Report the highest Ratio you can.
"""
import sys, math, ast, random

SEED_BASE = 882000
X_LO = 5.0
BAND_HI_EXTRA = 15.0
N_TRAIN = 90
NOISE_SIGMA = 0.025

HELDOUT_SIGMA = 0.05          # held-out observation-noise floor (irreducible)
LAMBDA = 0.010
GAMMA = 0.18
SCORE_CAP = 0.90
MAX_NODES = 200
MAX_OUT_BYTES = 200000
CLAMP_LOG = 80.0

_PLAN_ALPHA = {1: 1.35, 2: 1.55, 3: 3.55, 4: 3.25, 5: 2.05,
               6: 1.45, 7: 3.75, 8: 2.55, 9: 3.05, 10: 1.85}

ALLOWED_FUNCS_ARITY = {"absv": 1, "powv": 2}

_HELDOUT_OFFSETS = [40, 60, 90, 130, 180, 240, 310, 390, 480, 580]
_HELDOUT_CORNERS = [700, 850, 1000, 1200, 1450]


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden law (IDENTICAL to gen.py) ----------
def hidden_law(t):
    rng = random.Random(SEED_BASE + t * 7919)
    xc = rng.uniform(12.0, 30.0)
    a0 = rng.uniform(1.0, 3.0)
    a1 = rng.uniform(0.15, 0.45)
    a2 = rng.uniform(0.004, 0.018)
    alpha = _PLAN_ALPHA.get(t, 2.20) + rng.uniform(-0.05, 0.05)
    B = rng.uniform(0.6, 2.2)
    return xc, a0, a1, a2, alpha, B


def y_true(x, xc, a0, a1, a2, alpha, B):
    if x < xc:
        return a0 + a1 * x + a2 * x * x
    y0 = a0 + a1 * xc + a2 * xc * xc
    return y0 + B * ((x - xc) ** alpha)


def train_rows(t):
    xc, a0, a1, a2, alpha, B = hidden_law(t)
    hi = xc + BAND_HI_EXTRA
    rng = random.Random(40410 + t * 13)
    rows = []
    for i in range(N_TRAIN):
        frac = (i + rng.uniform(0.05, 0.95)) / N_TRAIN
        frac = min(0.999999, max(0.000001, frac))
        x = X_LO + frac * (hi - X_LO)
        clean = y_true(x, xc, a0, a1, a2, alpha, B)
        noisy = clean * math.exp(rng.gauss(0.0, NOISE_SIGMA))
        rows.append((x, noisy))
    rows.sort(key=lambda r: r[0])
    return rows


def heldout(t):
    xc, a0, a1, a2, alpha, B = hidden_law(t)
    rng = random.Random(20260 + t * 29)
    offs = _HELDOUT_OFFSETS + _HELDOUT_CORNERS
    clean = []
    noisy = []
    for off in offs:
        x = xc + off
        c = y_true(x, xc, a0, a1, a2, alpha, B)
        n = c * math.exp(rng.gauss(0.0, HELDOUT_SIGMA))
        clean.append((x, c))
        noisy.append((x, n))
    return clean, noisy


# ---------- expression parsing / validation ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
)


def _validate_ast(tree):
    used = set()
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return None, "disallowed syntax %s" % type(node).__name__
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS_ARITY):
                return None, "disallowed call"
            if node.keywords:
                return None, "keyword args not allowed"
            need = ALLOWED_FUNCS_ARITY[node.func.id]
            if len(node.args) != need:
                return None, "%s takes %d arg(s)" % (node.func.id, need)
        if isinstance(node, ast.Name):
            nm = node.id
            if nm in ALLOWED_FUNCS_ARITY:
                continue
            if nm != "x":
                return None, "unknown name %s" % nm
            used.add(nm)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                return None, "non-numeric constant"
            v = float(node.value)
            if v != v or v in (float("inf"), float("-inf")):
                return None, "non-finite constant"
    return used, None


def _count_nodes(tree):
    return sum(1 for nd in ast.walk(tree)
               if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant)))


def parse_law(raw):
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        fail("empty output")
    text = lines[-1]
    low = text.lower()
    for pre in ("y", "y(x)", "out", "f(x)"):
        if low.startswith(pre + "=") or low.startswith(pre + " ="):
            text = text[len(pre):].strip()
            break
    if text.startswith("="):
        text = text[1:].strip()
    if not text:
        fail("empty expression")
    if len(text) > MAX_OUT_BYTES:
        fail("expression too long")
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    used, err = _validate_ast(tree)
    if err:
        fail(err)
    nodes = _count_nodes(tree)
    if nodes > MAX_NODES:
        fail("law too large (%d nodes)" % nodes)
    try:
        code = compile(tree, "<law>", "eval")
    except Exception:
        fail("compile error")
    return code, nodes


def _powv(a, b):
    if not isinstance(a, (int, float)) or isinstance(a, bool):
        return float("nan")
    if not isinstance(b, (int, float)) or isinstance(b, bool):
        return float("nan")
    if a <= 0.0:
        return float("nan")
    try:
        return math.pow(a, b)
    except Exception:
        return float("nan")


_FUNCS = {
    "absv": abs,
    "powv": _powv,
}


def eval_law(code, x):
    env = dict(_FUNCS)
    env["x"] = float(x)
    try:
        v = eval(code, {"__builtins__": {}}, env)
    except Exception:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            header = fh.readline().split()
        t = int(header[0])
    except Exception:
        fail("bad instance header")
    if t < 1 or t > 100000:
        fail("bad test id")

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace")

    code, nodes = parse_law(text)

    rows = train_rows(t)
    ys = [y for (_, y) in rows]
    geo_log_mean = sum(math.log(y) for y in ys) / len(ys)

    clean, noisy = heldout(t)

    se = []
    for (x, _c), (_, ny) in zip(clean, noisy):
        pred = eval_law(code, x)
        if pred is None or pred <= 0.0:
            fail("non-finite/non-positive prediction at x=%.3g" % x)
        lp = max(-CLAMP_LOG, min(CLAMP_LOG, math.log(pred)))
        lt = math.log(ny)
        se.append((lp - lt) ** 2)
    F_mse = sum(se) / len(se)

    se_b = [(geo_log_mean - math.log(ny)) ** 2 for (_, ny) in noisy]
    B_mse = sum(se_b) / len(se_b)

    F = F_mse * (1.0 + LAMBDA * nodes)
    Bs = B_mse * (1.0 + LAMBDA * 1)
    ratio_raw = Bs / max(1e-9, F)
    sc = min(SCORE_CAP, 0.1 * (ratio_raw ** GAMMA))
    print("heldout_MSLE=%.6f baseline_MSLE=%.6f nodes=%d B/F=%.4f  Ratio: %.6f"
          % (F_mse, B_mse, nodes, ratio_raw, sc))


if __name__ == "__main__":
    main()
