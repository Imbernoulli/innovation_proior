#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the mystery-converter loss-law recovery task. The
solver submits ONE closed-form expression for the power loss y as a function
of load fraction L and ambient temperature T.

- Reads the case id from <in> (header), then regenerates the hidden loss law
  and the HELD-OUT OVERLOAD grid (L in [0.80,1.10], never seen in training)
  entirely from that id. The law and its coefficients live ONLY here.
- Parses the submitted expression with a strict AST whitelist:
      names     L T
      operators + - * / **  and unary +/-
      functions sqrt log exp sig tanh absv
      numeric constants
- Evaluates it on the held-out grid, computes a bounded symmetric relative
  error per point, averages, and adds a small node-count parsimony penalty
  (minimise):
      metric = mean_i min(1, |p_i - t_i| / (|p_i| + |t_i|))
      O = metric * (1 + LAMBDA * nodes)
      B = baseline_metric * (1 + LAMBDA * 1)   # baseline = constant mean(train y)
      Ratio = min(1000, 100 * B / O) / 1000
  A constant reproduces the baseline (~0.1). A law built only from the two
  "obvious" mechanisms (standby + temperature-modulated resistive loss) or
  from a generically flexible low-order polynomial in L cannot express the
  faster-than-quadratic core-saturation growth and drifts increasingly wrong
  as the held-out grid moves past rated load. Held-out noise (larger than
  training noise -- overload readings are less controlled) keeps even a
  correctly-shaped law well below the ceiling, leaving headroom.
"""
import sys, math, ast, random

# ---- fixed design constants (mirrored byte-for-byte in gen.py) ----
L_TR_LO, L_TR_HI = 0.20, 0.60
T_LO, T_HI = -10.0, 50.0
N_TRAIN = 200
NOISE_TRAIN = 0.025

# ---- held-out / scoring constants (grader only) ----
L_HE_LO, L_HE_HI = 0.80, 1.10
N_HELD = 140
NOISE_HELD = 0.22
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
ALLOWED_NAMES = {"L", "T"}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden loss law (identical to gen.py) ----------
def params(t):
    rng = random.Random(500000 + t * 91013)
    p_exp = rng.uniform(3.4, 5.6)
    P0 = rng.uniform(3.0, 9.0)
    c1 = rng.uniform(45.0, 95.0)
    beta = rng.uniform(-0.006, -0.002)
    c2 = c1 * beta
    ratio1 = rng.uniform(0.45, 0.85)
    ks = ratio1 * c1
    return p_exp, P0, c1, c2, ks


def true_y(L, T, p_exp, P0, c1, c2, ks):
    return P0 + c1 * L * L + c2 * T * L * L + ks * L ** p_exp


def gen_train(t):
    p_exp, P0, c1, c2, ks = params(t)
    rng = random.Random(1000 + t * 13)
    rows = []
    for _ in range(N_TRAIN):
        L = rng.uniform(L_TR_LO, L_TR_HI)
        T = rng.uniform(T_LO, T_HI)
        y = true_y(L, T, p_exp, P0, c1, c2, ks) * math.exp(rng.gauss(0.0, NOISE_TRAIN))
        rows.append((L, T, y))
    return rows


def gen_held(t):
    """Held-out OVERLOAD grid: L far above the training band."""
    p_exp, P0, c1, c2, ks = params(t)
    rng = random.Random(9000 + t * 7)
    pts = []
    for _ in range(N_HELD):
        L = rng.uniform(L_HE_LO, L_HE_HI)
        T = rng.uniform(T_LO, T_HI)
        y = true_y(L, T, p_exp, P0, c1, c2, ks) * math.exp(rng.gauss(0.0, NOISE_HELD))
        pts.append((L, T, y))
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


def eval_at(code, L, T):
    env = dict(ALLOWED_FUNCS)
    env["L"] = L; env["T"] = T
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
    for L, T, y in held:
        p = eval_at(code, L, T)
        if p is None:
            fail("non-finite / invalid prediction")
        d = abs(p - y) / (abs(p) + abs(y) + 1e-30)
        ds.append(min(CAP, d))
    metric = sum(ds) / len(ds)

    # baseline: constant predictor = mean of TRAIN y
    train = gen_train(t)
    mn = sum(r[2] for r in train) / len(train)
    bd = [min(CAP, abs(mn - y) / (abs(mn) + abs(y) + 1e-30)) for _, _, y in held]
    Bmetric = sum(bd) / len(bd)

    B = Bmetric * (1.0 + LAMBDA * 1)
    O = metric * (1.0 + LAMBDA * nodes)
    sc = min(1000.0, 100.0 * B / max(1e-12, O))
    print("metric=%.6f baseline=%.6f nodes=%d  Ratio: %.6f"
          % (metric, Bmetric, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
