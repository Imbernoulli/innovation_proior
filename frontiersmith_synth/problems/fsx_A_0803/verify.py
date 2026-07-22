#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the magnet-near-Tc scaling-collapse task.  The
solver submits ONE closed-form expression for the order parameter m as a
function of the temperature T and the applied field h.

- Reads the case id from <in> (header), then regenerates the hidden critical
  law and the HELD-OUT EXTRAPOLATION grid (mostly MUCH closer to Tc than any
  training point, on BOTH sides of the transition) entirely from that id.
  The law and its coefficients live ONLY here.
- Parses the submitted expression with a strict AST whitelist:
      names     T h
      operators + - * / **  and unary +/-
      functions sqrt log exp sig tanh absv
      numeric constants
- Evaluates it on the held-out grid, computes a bounded symmetric relative
  error per point, averages, and adds a small node-count parsimony penalty
  (minimise):
      metric = mean_i min(1, |p_i - t_i| / (|p_i| + |t_i|))
      O = metric * (1 + LAMBDA * nodes)
      B = baseline_metric * (1 + LAMBDA * 1)   # baseline = constant geomean(train m)
      Ratio = min(1000, 100 * B / O) / 1000
  A constant reproduces the baseline (~0.1).  A generic smooth (polynomial /
  log-linear) fit of m against T and log(h) matches the training band, but the
  true law is NON-ANALYTIC at Tc (|Tc-T|^beta, beta not an integer) and folds
  onto |T-Tc| on the far side -- a smooth extrapolation cannot reproduce
  either feature and stays far off on the held-out grid.  Only a predictor
  that finds the SAME change of variables (Tc, phi) under which every h-curve
  collapses onto one curve F(x) = (1+x^2)^(-p) survives extrapolation; held-out
  noise keeps even a correct collapse well below the ceiling, leaving headroom.
"""
import sys, math, ast, random

# ---- fixed design constants (mirrored byte-for-byte in gen.py) ----
USWEEP = math.log(1.6)
HSWEEP = math.log(15.0)
NOISE_TRAIN = 0.04
N_TRAIN = 180

# ---- held-out / scoring constants (grader only) ----
NOISE_HELD = 0.10
N_HELD = 140
SHRINK_LO = math.log(3.0)     # held-out "near Tc" points sit 3x..20x CLOSER to Tc
SHRINK_HI = math.log(20.0)    #   than the typical training distance
NEAR_PROB = 0.52               # fraction of held-out points forced close to Tc (either side)
FAR_JITTER = math.log(1.4)     # remaining points stay near the training-typical distance
LAMBDA = 0.004
CAP = 1.0
MAX_NODES = 60
MAX_OUT_BYTES = 100000

ALLOWED_FUNCS = {
    "sqrt": lambda x: math.sqrt(x),
    "log":  lambda x: math.log(x),
    "exp":  lambda x: math.exp(max(-700.0, min(700.0, x))),
    "sig":  lambda x: 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x)))),
    "tanh": math.tanh,
    "absv": abs,
}
ALLOWED_NAMES = {"T", "h"}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden critical law (identical to gen.py) ----------
def params(t):
    rng = random.Random(511000 + t * 8123457)
    Tc = rng.uniform(1.60, 2.40)
    beta = rng.uniform(0.25, 0.55)
    phi = rng.uniform(1.00, 2.20)
    A = rng.uniform(0.80, 1.60)
    U0 = math.exp(rng.uniform(math.log(0.35), math.log(0.55)))
    H0 = math.exp(rng.uniform(math.log(0.05), math.log(0.40)))
    return Tc, beta, phi, A, U0, H0


def true_m(T, h, Tc, beta, phi, A):
    u = abs(Tc - T)
    if u < 1e-9:
        u = 1e-9
    x = h / (u ** phi)
    p = beta / (2.0 * phi)
    F = (1.0 + x * x) ** (-p)
    return A * (u ** beta) * F


def gen_train(t):
    Tc, beta, phi, A, U0, H0 = params(t)
    rng = random.Random(191 + t * 17)
    rows = []
    for _ in range(N_TRAIN):
        u = U0 * math.exp(rng.uniform(-USWEEP, USWEEP))
        T = Tc - u
        h = H0 * math.exp(rng.uniform(-HSWEEP, HSWEEP))
        m = true_m(T, h, Tc, beta, phi, A) * math.exp(rng.gauss(0.0, NOISE_TRAIN))
        rows.append((T, h, m))
    return rows


def gen_held(t):
    """Held-out grid: NEAR_PROB fraction is hard extrapolation (much closer to
    Tc, either side, including the never-seen T > Tc side); the rest is a fair
    same-side interpolation check (distance close to the training-typical
    band, fresh h values) so a locally-good fit is rewarded for that alone."""
    Tc, beta, phi, A, U0, H0 = params(t)
    rng = random.Random(90210 + t * 31)
    pts = []
    for _ in range(N_HELD):
        if rng.random() < NEAR_PROB:
            shrink = math.exp(rng.uniform(SHRINK_LO, SHRINK_HI))
            side = 1.0 if rng.random() < 0.5 else -1.0   # either side of Tc
        else:
            shrink = math.exp(rng.uniform(-FAR_JITTER, FAR_JITTER))
            side = 1.0                                    # same side as training
        u = U0 / shrink
        T = Tc - side * u
        h = H0 * math.exp(rng.uniform(-HSWEEP, HSWEEP))
        m = true_m(T, h, Tc, beta, phi, A) * math.exp(rng.gauss(0.0, NOISE_HELD))
        pts.append((T, h, m))
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
    text = lines[-1]                       # take the last non-empty line
    if text.upper().startswith("EXPR "):   # optional leading tag
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


def eval_at(code, T, h):
    env = dict(ALLOWED_FUNCS)
    env["T"] = T; env["h"] = h
    try:
        p = eval(code, {"__builtins__": {}}, env)
    except Exception:
        return None
    if isinstance(p, bool) or not isinstance(p, (int, float)):
        return None                        # rejects complex results from neg**frac
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
        t = int(header[1])
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

    code, nodes = parse_expr(text)

    held = gen_held(t)
    ds = []
    for T, h, m in held:
        p = eval_at(code, T, h)
        if p is None:
            fail("non-finite / invalid prediction")
        d = abs(p - m) / (abs(p) + abs(m) + 1e-30)
        ds.append(min(CAP, d))
    metric = sum(ds) / len(ds)

    # baseline: constant predictor = geometric mean of TRAIN m
    train = gen_train(t)
    gm = math.exp(sum(math.log(r[2]) for r in train) / len(train))
    bd = [min(CAP, abs(gm - m) / (abs(gm) + abs(m) + 1e-30)) for _, _, m in held]
    Bmetric = sum(bd) / len(bd)

    B = Bmetric * (1.0 + LAMBDA * 1)
    O = metric * (1.0 + LAMBDA * nodes)
    sc = min(1000.0, 100.0 * B / max(1e-12, O))
    print("metric=%.6f baseline=%.6f nodes=%d  Ratio: %.6f"
          % (metric, Bmetric, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
