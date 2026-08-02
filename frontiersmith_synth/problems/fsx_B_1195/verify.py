#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the bridge-settlement forecast task. The solver submits
ONE closed-form expression for displacement d as a function of elapsed days t and
deck temperature T.

- Reads the case id from <in> (header only), then regenerates the hidden law and
  the HELD-OUT / EXTRAPOLATION horizon (t several seasonal cycles beyond the
  visible span, NEVER seen in training) entirely from that id. The law and its
  coefficients live ONLY here.
- Parses the submitted expression with a strict AST whitelist:
      names     t T
      operators + - * / **  and unary +/-
      functions sqrt log exp sig tanh absv
      numeric constants
- Evaluates it on the held-out horizon, computes a bounded symmetric relative
  error per point, averages, and adds a small node-count parsimony penalty
  (minimise):
      metric = mean_i min(1, |p_i - t_i| / (|p_i| + |t_i|))
      O = metric * (1 + LAMBDA * nodes)
      B = baseline_metric * (1 + LAMBDA * 1)   # baseline = constant mean(train d)
      Ratio = min(1000, 100 * B / O) / 1000
  A constant reproduces the baseline (~0.1). An expression that treats the whole
  visible signal (irreversible settlement + reversible seasonal wobble, both
  present in EVERY reading) as if it were a single undecomposed trend inherits
  whatever seasonal phase the visible window happened to end on -- for the
  trap cases (window truncated at a seasonal peak) that phase contamination
  gets extrapolated forward across several more cycles and drifts far from the
  truth. Held-out sensor + process noise (larger than in training) keeps even a
  correctly-shaped law below the ceiling, leaving headroom.
"""
import sys, math, ast, random

# ---- fixed design constants (mirrored byte-for-byte in gen.py) ----
P = 365.0
T_END_VIS = 1200.0
N_TRAIN = 200
TRAP_IDS = {1, 2, 3, 4, 5, 6, 7}

# ---- held-out / scoring constants (grader only) ----
HELD_LO = T_END_VIS + 3.0 * P
HELD_HI = T_END_VIS + 5.0 * P
N_HELD = 150
HELD_NOISE_T_MULT = 1.15
HELD_NOISE_D_MULT = 1.6
LAMBDA = 0.003
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
ALLOWED_NAMES = {"t", "T"}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden displacement law (identical to gen.py) ----------
def params(t):
    rng = random.Random(700000 + t * 104729)
    Dinf = rng.uniform(35.0, 65.0)
    tau = rng.uniform(500.0, 950.0)
    drift = rng.uniform(0.01, 0.025)
    alpha = rng.uniform(0.9, 1.8)
    Tmean = rng.uniform(8.0, 18.0)
    Tamp = rng.uniform(8.0, 15.0)
    if t in TRAP_IDS:
        phase = (T_END_VIS - rng.uniform(-5.0, 5.0)) % P
    else:
        phase = rng.uniform(0.0, P)
    sigma_T = rng.uniform(1.2, 2.2)
    sigma_d = rng.uniform(4.5, 8.0)
    return Dinf, tau, drift, alpha, Tmean, Tamp, phase, sigma_T, sigma_d


def T_true(t, Tmean, Tamp, phase):
    return Tmean + Tamp * math.cos(2.0 * math.pi * (t - phase) / P)


def d_creep(t, Dinf, tau, drift):
    return Dinf * (1.0 - math.exp(-t / tau)) + drift * t


def d_thermal(t, alpha, Tmean, Tamp, phase):
    return alpha * (T_true(t, Tmean, Tamp, phase) - Tmean)


def gen_train(t):
    Dinf, tau, drift, alpha, Tmean, Tamp, phase, sigma_T, sigma_d = params(t)
    rng = random.Random(1000 + t * 13)
    rows = []
    for i in range(N_TRAIN):
        tt = i * (T_END_VIS / N_TRAIN)
        Traw = T_true(tt, Tmean, Tamp, phase) + rng.gauss(0.0, sigma_T)
        d = (d_creep(tt, Dinf, tau, drift)
             + d_thermal(tt, alpha, Tmean, Tamp, phase)
             + rng.gauss(0.0, sigma_d))
        rows.append((tt, Traw, d))
    return rows


def gen_held(t):
    """Held-out horizon: t several seasonal cycles beyond the visible span."""
    Dinf, tau, drift, alpha, Tmean, Tamp, phase, sigma_T, sigma_d = params(t)
    rng = random.Random(9000 + t * 7)
    sigma_T_h = sigma_T * HELD_NOISE_T_MULT
    sigma_d_h = sigma_d * HELD_NOISE_D_MULT
    pts = []
    for _ in range(N_HELD):
        tt = HELD_LO + rng.uniform(0.0, HELD_HI - HELD_LO)
        Traw = T_true(tt, Tmean, Tamp, phase) + rng.gauss(0.0, sigma_T_h)
        dtrue = (d_creep(tt, Dinf, tau, drift)
                 + d_thermal(tt, alpha, Tmean, Tamp, phase)
                 + rng.gauss(0.0, sigma_d_h))
        pts.append((tt, Traw, dtrue))
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


def eval_at(code, t, T):
    env = dict(ALLOWED_FUNCS)
    env["t"] = t; env["T"] = T
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
        cid = int(header[1])
    except Exception:
        fail("bad instance header")
    if cid < 1 or cid > 100000:
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

    held = gen_held(cid)
    ds = []
    for tt, Traw, dtrue in held:
        p = eval_at(code, tt, Traw)
        if p is None:
            fail("non-finite / invalid prediction")
        e = abs(p - dtrue) / (abs(p) + abs(dtrue) + 1e-30)
        ds.append(min(CAP, e))
    metric = sum(ds) / len(ds)

    # baseline: constant predictor = mean of TRAIN d (regenerated internally, not
    # re-parsed from <in> -- <in> is only used to recover the case id)
    train = gen_train(cid)
    mn = sum(r[2] for r in train) / len(train)
    bd = [min(CAP, abs(mn - dtrue) / (abs(mn) + abs(dtrue) + 1e-30)) for _, _, dtrue in held]
    Bmetric = sum(bd) / len(bd)

    B = Bmetric * (1.0 + LAMBDA * 1)
    O = metric * (1.0 + LAMBDA * nodes)
    sc = min(1000.0, 100.0 * B / max(1e-12, O))
    print("metric=%.6f baseline=%.6f nodes=%d  Ratio: %.6f"
          % (metric, Bmetric, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
