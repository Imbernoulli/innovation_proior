#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the cache-miss capacity-cliff forecast task.  The
solver submits ONE closed-form Python expression predicting the miss rate as
a function of the working-set size `n`.

- Reads the test id (and public C, A) from <in>, then regenerates the hidden
  hardware/workload law and a HELD-OUT set of working-set sizes -- all well
  past the training range, straddling and crossing the capacity cliff --
  entirely from that id.  The shape parameter and the exact crossing law
  live ONLY here (and are re-derived deterministically, never imported).
- Parses the submitted expression with a strict AST whitelist:
      names     n
      operators + - * / **  and unary +/-
      functions sqrt log exp sig tanh absv
      numeric constants
- Evaluates it on the held-out sizes, computes a clipped absolute error
  against the (noisy, finite-sample) true miss rate, averages, and adds a
  small node-count parsimony penalty (minimise):
      metric = mean_i min(1, |p_i - t_i|)
      O = metric * (1 + LAMBDA * nodes)
      B = baseline_metric * (1 + LAMBDA * 1)   # baseline = constant train-mean
      Ratio = min(1000, 100 * B / O) / 1000
  A constant reproduces the baseline (~0.1).  A smooth curve fit of the
  visible (N, missrate) log alone cannot see the cliff -- every training row
  sits deep in the sub-cliff regime, where the true curve is a tiny power-law
  tail easily swamped by finite-sample noise, so it looks flat/near-zero no
  matter the true steepness.  Only a predictor that reads the reuse-distance
  HISTOGRAM (which reveals the steepness directly, independent of scale) and
  combines it with the PUBLIC effective-capacity law K = C*A/(A+1) can place
  the cliff and its shape correctly; finite-sample noise on the held-out set
  keeps even a correct law below the ceiling, leaving headroom.
"""
import sys, math, ast, random

# ---- fixed design constants (mirrored byte-for-byte in gen.py) ----
C_LO, C_HI = 800, 30000
A_CHOICES = [1, 2, 4, 8, 16, 32]
S_LO, S_HI = 1.3, 4.6
M_HIST = 60
N_TRAIN = 14
TRAIN_FRAC_LO, TRAIN_FRAC_HI = 0.01, 0.15
TTRIAL_TRAIN = 6000

# ---- held-out / scoring constants (grader only) ----
HELD_FRACS = [0.25, 0.45, 0.7, 1.0, 1.4, 2.2, 4.0, 9.0, 20.0]
TTRIAL_HELD = 10
LAMBDA = 0.01
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
ALLOWED_NAMES = {"n"}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden law (identical to gen.py) ----------
def params(t):
    rng = random.Random(3010001 + t * 800213)
    C = rng.randint(C_LO, C_HI)
    A = rng.choice(A_CHOICES)
    s = rng.uniform(S_LO, S_HI)
    K = C * A / (A + 1.0)
    return C, A, s, K


def true_missrate(N, K, s):
    r = (N / K) ** s
    return r / (1.0 + r)


def gen_train(t, K, s):
    rng = random.Random(2290007 + t * 60013)
    rows = []
    for _ in range(N_TRAIN):
        frac = math.exp(rng.uniform(math.log(TRAIN_FRAC_LO), math.log(TRAIN_FRAC_HI)))
        N = max(1.0, K * frac)
        p = true_missrate(N, K, s)
        misses = 0
        for _ in range(TTRIAL_TRAIN):
            if rng.random() < p:
                misses += 1
        rows.append((N, misses / float(TTRIAL_TRAIN)))
    return rows


def gen_held(t, K, s):
    """Held-out working-set sizes straddling and crossing the capacity
    cliff -- genuine extrapolation past every training N (frac <= 0.15)."""
    rng = random.Random(8810009 + t * 4001)
    pts = []
    for frac in HELD_FRACS:
        N = K * frac
        p = true_missrate(N, K, s)
        misses = 0
        for _ in range(TTRIAL_HELD):
            if rng.random() < p:
                misses += 1
        pts.append((N, misses / float(TTRIAL_HELD)))
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


def eval_at(code, n_val):
    env = dict(ALLOWED_FUNCS)
    env["n"] = n_val
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
        t = int(header[0])
    except Exception:
        fail("bad instance header")
    if t < 1 or t > 100000:
        fail("bad test id")

    C, A, s, K = params(t)

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace")

    code, nodes = parse_expr(text)

    held = gen_held(t, K, s)
    ds = []
    for N, mr in held:
        p = eval_at(code, N)
        if p is None:
            fail("non-finite / invalid prediction")
        ds.append(min(CAP, abs(p - mr)))
    metric = sum(ds) / len(ds)

    # baseline: constant predictor = mean of TRAIN missrate
    train = gen_train(t, K, s)
    cm = sum(r[1] for r in train) / len(train)
    bd = [min(CAP, abs(cm - mr)) for _, mr in held]
    Bmetric = sum(bd) / len(bd)

    B = Bmetric * (1.0 + LAMBDA * 1)
    O = metric * (1.0 + LAMBDA * nodes)
    sc = min(1000.0, 100.0 * B / max(1e-12, O))
    print("metric=%.6f baseline=%.6f nodes=%d  Ratio: %.6f"
          % (metric, Bmetric, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
