#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the assembly stack-up variance-composition recovery
task. The solver submits ONE closed-form expression for the total deviation D
as a function of the four visible structural summaries n, m, S2, B2.

- Reads the case id from <in> (header), then regenerates the hidden law and
  the HELD-OUT EXTRAPOLATION log (much longer assemblies, shipped in a
  handful of LARGE batches, so B2 grows quadratically with n instead of
  linearly) entirely from that id. The law and its coefficients live ONLY
  here.
- Parses the submitted expression with a strict AST whitelist:
      names     n m S2 B2
      operators + - * / **  and unary +/-
      functions sqrt log exp sig tanh absv
      numeric constants
- Evaluates it on the held-out log, computes a bounded symmetric relative
  error per row, averages, and adds a small node-count parsimony penalty
  (minimise):
      metric = mean_i min(1, |p_i - t_i| / (|p_i| + |t_i|))
      O = metric * (1 + LAMBDA * nodes)
      B = baseline_metric * (1 + LAMBDA * 1)   # baseline = constant geomean(train D)
      Ratio = min(1000, 100 * B / O) / 1000
  A constant reproduces the baseline (~0.1). A generic power-law / RSS-style
  fit to the pilot-line log matches training well -- the correlated term
  TAU2*B2 is a small fraction of S2 there, comparable to sensor noise -- but
  it structurally never isolates a separate additive term keyed on B2, so on
  the held-out log (large batches, B2 ~ n^2) it diverges. Only a predictor
  that hypothesizes the additive-under-sqrt composition D = sqrt(S2 + c*B2)
  and fits c from the residual (D^2 - S2) extrapolates; held-out noise keeps
  even a correct law below the ceiling, leaving headroom.
"""
import sys, math, ast, random

# ---- fixed design constants (mirrored byte-for-byte in gen.py) ----
SEED_LAW_BASE   = 800000
SEED_TRAIN_BASE = 211
SPREAD          = math.log(1.4)
N_TRAIN_LO, N_TRAIN_HI = 3, 12
NOISE_TRAIN     = 0.04
N_TRAIN         = 160

# ---- held-out / scoring constants (grader only) ----
SEED_HELD_BASE  = 500009
N_HELD_LO, N_HELD_HI = 50, 220
M_HELD_LO, M_HELD_HI = 2, 5
NOISE_HELD      = 0.35
N_HELD          = 120
LAMBDA          = 0.004
CAP             = 1.0
MAX_NODES       = 60
MAX_OUT_BYTES   = 100000

ALLOWED_FUNCS = {
    "sqrt": lambda x: math.sqrt(x),
    "log":  lambda x: math.log(x),
    "exp":  lambda x: math.exp(max(-700.0, min(700.0, x))),
    "sig":  lambda x: 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x)))),
    "tanh": math.tanh,
    "absv": abs,
}
ALLOWED_NAMES = {"n", "m", "S2", "B2"}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden variance-composition law (identical to gen.py) ----------
def params(t):
    rng = random.Random(SEED_LAW_BASE + t * 9176111)
    SIG0 = math.exp(rng.uniform(math.log(0.4), math.log(4.0)))
    FRAC = rng.uniform(0.01, 0.06)
    TAU2 = SIG0 * SIG0 * FRAC
    return SIG0, TAU2


def batches_train(rng, n):
    sizes = []
    rem = n
    while rem > 0:
        b = min(rem, rng.randint(1, 3))
        sizes.append(b)
        rem -= b
    return sizes


def gen_sigmas(rng, n, SIG0):
    return [SIG0 * math.exp(rng.uniform(-SPREAD, SPREAD)) for _ in range(n)]


def true_D(S2, B2, TAU2):
    return math.sqrt(S2 + TAU2 * B2)


def gen_train(t):
    SIG0, TAU2 = params(t)
    rng = random.Random(SEED_TRAIN_BASE + t * 13)
    rows = []
    for _ in range(N_TRAIN):
        n = rng.randint(N_TRAIN_LO, N_TRAIN_HI)
        sizes = batches_train(rng, n)
        m = len(sizes)
        sigmas = gen_sigmas(rng, n, SIG0)
        S2 = sum(s * s for s in sigmas)
        B2 = float(sum(b * b for b in sizes))
        Dtrue = true_D(S2, B2, TAU2)
        D = Dtrue * math.exp(rng.gauss(0.0, NOISE_TRAIN))
        rows.append((n, m, S2, B2, D))
    return rows


def batches_held(rng, n):
    """Production line: a HANDFUL of LARGE batches -- batch count stays O(1)
    as n grows, so each batch's size (and B2) scales up with n."""
    m = rng.randint(M_HELD_LO, M_HELD_HI)
    base = n // m
    sizes = [base] * m
    leftover = n - base * m
    order = list(range(m))
    rng.shuffle(order)
    for i in range(leftover):
        sizes[order[i]] += 1
    return [s for s in sizes if s > 0]


def gen_held(t):
    """Held-out extrapolation log: much longer assemblies, large-lot batching."""
    SIG0, TAU2 = params(t)
    rng = random.Random(SEED_HELD_BASE + t * 7)
    rows = []
    for _ in range(N_HELD):
        n = rng.randint(N_HELD_LO, N_HELD_HI)
        sizes = batches_held(rng, n)
        m = len(sizes)
        sigmas = gen_sigmas(rng, n, SIG0)
        S2 = sum(s * s for s in sigmas)
        B2 = float(sum(b * b for b in sizes))
        Dtrue = true_D(S2, B2, TAU2)
        D = Dtrue * math.exp(rng.gauss(0.0, NOISE_HELD))
        rows.append((n, m, S2, B2, D))
    return rows


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


def eval_at(code, n, m, S2, B2):
    env = dict(ALLOWED_FUNCS)
    env["n"] = n; env["m"] = m; env["S2"] = S2; env["B2"] = B2
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
    for n, m, S2, B2, D in held:
        p = eval_at(code, n, m, S2, B2)
        if p is None:
            fail("non-finite / invalid prediction")
        d = abs(p - D) / (abs(p) + abs(D) + 1e-30)
        ds.append(min(CAP, d))
    metric = sum(ds) / len(ds)

    # baseline: constant predictor = geometric mean of TRAIN D
    train = gen_train(t)
    gm = math.exp(sum(math.log(r[4]) for r in train) / len(train))
    bd = [min(CAP, abs(gm - D) / (abs(gm) + abs(D) + 1e-30)) for _, _, _, _, D in held]
    Bmetric = sum(bd) / len(bd)

    B = Bmetric * (1.0 + LAMBDA * 1)
    O = metric * (1.0 + LAMBDA * nodes)
    sc = min(1000.0, 100.0 * B / max(1e-12, O))
    print("metric=%.6f baseline=%.6f nodes=%d  Ratio: %.6f"
          % (metric, Bmetric, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
