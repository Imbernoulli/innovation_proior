#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the recsys organic-vs-induced popularity-forecast
task. The solver submits ONE closed-form Python expression predicting an
item's engagement rate as a function of the period `t` and the exposure
fraction `x` it would receive that period.

- Reads the test id from <in>, then regenerates the hidden organic/exposure
  law (organic baseline O0, organic drift O1, induced-per-exposure lift
  ALPHA, feedback gain BETA, exogenous-shock gain GAMMA) and a HELD-OUT set
  of (period, exposure, engagement) triples entirely from that id. The
  held-out period range is genuine EXTRAPOLATION strictly past every
  training period, AND the exposure at each held-out point is drawn
  INDEPENDENTLY of history -- an "intervention" that breaks the
  exposure-feedback loop the training log was generated under (the
  recommender's own adaptive policy is switched off / randomized, exactly
  as a platform holdout/interleaving experiment would do). The shape
  parameters live ONLY here (and are re-derived deterministically, never
  imported from gen.py).
- Parses the submitted expression with a strict AST whitelist:
      names     t x
      operators + - * / **  and unary +/-
      functions sqrt log exp sig tanh absv
      numeric constants
- Evaluates it on the held-out (t, x) pairs, computes a clipped absolute
  error against the (noisy, finite-sample) true engagement, averages, and
  adds a small node-count parsimony penalty (minimise):
      metric = mean_i min(CAP, |p_i - e_i|)
      O = metric * (1 + LAMBDA * nodes)
      B = baseline_metric * (1 + LAMBDA * 1)   # baseline = constant train-mean
      Ratio = min(1000, 100 * B / O) / 1000
  A constant reproduces the baseline (~0.1). A curve fit to raw (t, e) alone,
  or to (x, e) alone, from the visible log cannot see this: across the
  logged window, exposure x(t) was itself driven by the item's own recent
  engagement (feedback loop), so x drifted upward together with t --
  whatever slope such a fit extracts is a blend of genuine organic drift AND
  the loop's self-reinforcing amplification, and extrapolates that blended,
  inflated slope arbitrarily far past the point where the loop (and its
  compounding) is switched off. Only a predictor that uses BOTH `t` and `x`
  as separate regressors -- regressing engagement out onto the exposure log
  it was actually shown, thereby isolating the organic component from the
  induced one -- generalizes correctly once exposure is set by intervention
  rather than by the recommender's own adaptive policy.
"""
import sys, math, ast, random

# ---- fixed design constants (mirrored byte-for-byte in gen.py) ----
T_TRAIN = 30
X0 = 0.15
XMIN, XMAX = 0.0, 1.0
O0_LO, O0_HI = 0.5, 3.0
O1_LO, O1_HI = 0.010, 0.045
ALPHA_LO, ALPHA_HI = 1.5, 3.6
BETA_LO, BETA_HI = 0.4, 0.85
GAMMA_LO, GAMMA_HI = 0.15, 0.35
NORM = 4.5
NOISE_SIGMA_E = 0.19

# ---- held-out / scoring constants (grader only) ----
HELD_OFFSETS = [5, 10, 15, 20, 25, 30, 35, 40]   # strictly past T_TRAIN
LAMBDA = 0.01
CAP = 3.0
MAX_NODES = 60
MAX_OUT_BYTES = 100000

ALLOWED_FUNCS = {
    "sqrt": lambda v: math.sqrt(v),
    "log":  lambda v: math.log(v),
    "exp":  lambda v: math.exp(max(-700.0, min(700.0, v))),
    "sig":  lambda v: 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, v)))),
    "tanh": math.tanh,
    "absv": abs,
}
ALLOWED_NAMES = {"t", "x"}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


def _gauss(rng):
    u1 = max(1e-12, rng.random())
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


# ---------- hidden law (identical to gen.py) ----------
def params(t):
    rng = random.Random(4110001 + t * 700111)
    O0 = rng.uniform(O0_LO, O0_HI)
    O1 = rng.uniform(O1_LO, O1_HI)
    ALPHA = rng.uniform(ALPHA_LO, ALPHA_HI)
    BETA = rng.uniform(BETA_LO, BETA_HI)
    GAMMA = rng.uniform(GAMMA_LO, GAMMA_HI)
    return O0, O1, ALPHA, BETA, GAMMA


def gen_train(t, O0, O1, ALPHA, BETA, GAMMA):
    rng = random.Random(2290007 + t * 60013)
    rows = []
    e_prev = O0
    for tt in range(1, T_TRAIN + 1):
        z = rng.uniform(-1.0, 1.0)
        if tt == 1:
            x = X0 + GAMMA * z
        else:
            x = X0 + BETA * max(0.0, e_prev - O0) / NORM + GAMMA * z
        x = min(XMAX, max(XMIN, x))
        noise = NOISE_SIGMA_E * _gauss(rng)
        e = max(0.0, O0 + O1 * tt + ALPHA * x + noise)
        rows.append((tt, x, e))
        e_prev = e
    return rows


def gen_held(t, O0, O1, ALPHA):
    """Held-out (period, exposure, engagement) triples: periods strictly
    past every training period (genuine extrapolation), exposure drawn
    i.i.d. -- the feedback loop is broken by intervention/randomization,
    exactly as the statement promises."""
    rng = random.Random(9310007 + t * 4001)
    pts = []
    for off in HELD_OFFSETS:
        tt = T_TRAIN + off
        x = rng.uniform(XMIN, XMAX)
        noise = NOISE_SIGMA_E * _gauss(rng)
        e = max(0.0, O0 + O1 * tt + ALPHA * x + noise)
        pts.append((tt, x, e))
    return pts


# ---------- expression parsing / validation ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
    ast.USub, ast.UAdd,
)


def _validate(tree):
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return "disallowed syntax %s" % type(node).__name__
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS):
                return "disallowed call"
            if node.keywords or len(node.args) != 1:
                return "bad function arity"
        if isinstance(node, ast.Name):
            if node.id in ALLOWED_FUNCS or node.id in ALLOWED_NAMES:
                continue
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


def parse_expr(raw):
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        fail("empty submission")
    text = lines[-1]
    if text.upper().startswith("EXPR "):
        text = text[5:].strip()
    if not text:
        fail("empty expression")
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    err = _validate(tree)
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


def eval_at(code, t_val, x_val):
    env = dict(ALLOWED_FUNCS)
    env["t"] = t_val
    env["x"] = x_val
    try:
        p = eval(code, {"__builtins__": {}}, env)
    except Exception:
        return None
    if isinstance(p, bool) or not isinstance(p, (int, float)):
        return None
    p = float(p)
    if p != p or p in (float("inf"), float("-inf")):
        return None
    return p


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            header = fh.readline().split()
        tid = int(header[0])
    except Exception:
        fail("bad instance header")
    if tid < 1 or tid > 100000:
        fail("bad test id")

    O0, O1, ALPHA, BETA, GAMMA = params(tid)

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace")

    code, nodes = parse_expr(text)

    held = gen_held(tid, O0, O1, ALPHA)
    ds = []
    for tt, x, e in held:
        p = eval_at(code, float(tt), x)
        if p is None:
            fail("non-finite / invalid prediction")
        ds.append(min(CAP, abs(p - e)))
    metric = sum(ds) / len(ds)

    # baseline: constant predictor = mean of TRAIN engagement
    train = gen_train(tid, O0, O1, ALPHA, BETA, GAMMA)
    cm = sum(r[2] for r in train) / len(train)
    bd = [min(CAP, abs(cm - e)) for _, _, e in held]
    Bmetric = sum(bd) / len(bd)

    B = Bmetric * (1.0 + LAMBDA * 1)
    O = metric * (1.0 + LAMBDA * nodes)
    sc = min(1000.0, 100.0 * B / max(1e-12, O))
    print("metric=%.6f baseline=%.6f nodes=%d  Ratio: %.6f"
          % (metric, Bmetric, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
