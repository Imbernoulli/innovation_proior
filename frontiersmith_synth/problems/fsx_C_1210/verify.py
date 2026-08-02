#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the staggered-cohort novelty-decay recovery task.  The
solver submits ONE closed-form expression for the treatment lift as a function
of a single variable `age` (days since a cohort entered treatment).

- Reads the case id from <in> (header), then regenerates the hidden novelty-decay
  law and the HELD-OUT EXTRAPOLATION grid (ages far beyond anything visible in
  training, i.e. long after the two-week window that training covers) entirely
  from that id.  The law and its coefficients live ONLY here (never in gen.py's
  stdout, never in an importable module).
- Parses the submitted expression with a strict AST whitelist:
      names     age
      operators + - * / **  and unary +/-
      functions sqrt log exp sig tanh absv
      numeric constants
- Evaluates it at every held-out age, computes a bounded symmetric relative
  error against the true (noisy) held-out lift, averages, and adds a small
  node-count parsimony penalty (minimise):
      metric = mean_i min(1, |p_i - t_i| / (|p_i| + |t_i| + eps))
      O = metric * (1 + LAMBDA * nodes)
      B = baseline_metric * (1 + LAMBDA * 1)   # baseline = constant mean(train lift)
      Ratio = min(1000, 100 * B / O) / 1000
  A constant predictor reproduces the baseline (~0.1).  Averaging the visible
  window (or even just its most "mature" tail) still carries an un-removed
  fraction of the novelty spike and the common calendar wobble, so it stays
  well below what a fit that separates the decaying and persistent components
  achieves.  Held-out noise + an un-forecastable future calendar wobble keep
  even a correct decay/persistence split below the ceiling, leaving headroom.
"""
import sys, math, ast, random

# ---- fixed design constants (mirrored byte-for-byte in gen.py) ----
C_COHORTS = 6
GAP       = 6
W         = 16
T_VIS     = (C_COHORTS - 1) * GAP + W
PERIOD1   = T_VIS / 3.0
PERIOD2   = T_VIS / 5.0

# ---- held-out / scoring constants (grader only) ----
N_HELD    = 140
AGE_HELD_LO = W + 8.0     # held-out ages start well past the visible two-week tail
AGE_HELD_HI = W + 150.0   # ...and extend far into the persistent-lift regime
LAMBDA    = 0.01
CAP       = 1.0
MAX_NODES = 40
MAX_OUT_BYTES = 100000

ALLOWED_FUNCS = {
    "sqrt": lambda x: math.sqrt(x),
    "log":  lambda x: math.log(x),
    "exp":  lambda x: math.exp(max(-700.0, min(700.0, x))),
    "sig":  lambda x: 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x)))),
    "tanh": math.tanh,
    "absv": abs,
}
ALLOWED_NAMES = {"age"}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden novelty-decay law (identical to gen.py) ----------
def params(t):
    rng = random.Random(521000 + t * 8123911)
    P = rng.uniform(0.04, 0.10)
    ratio_lo, ratio_hi = 0.8 + 0.15 * t, 1.6 + 0.35 * t
    A = P * rng.uniform(ratio_lo, ratio_hi)
    tau_lo, tau_hi = 4.0 + 0.3 * t, 10.0 + 1.2 * t
    tau = rng.uniform(tau_lo, tau_hi)
    D_amp = P * rng.uniform(0.3, 0.9)
    phase1 = rng.uniform(0.0, 2 * math.pi)
    phase2 = rng.uniform(0.0, 2 * math.pi)
    sigma_train = rng.uniform(0.004, 0.010)
    sigma_held = sigma_train * rng.uniform(1.2, 1.6)
    return dict(P=P, A=A, tau=tau, D_amp=D_amp, phase1=phase1, phase2=phase2,
                sigma_train=sigma_train, sigma_held=sigma_held)


def calendar_wobble(t_cal, prm):
    d = (0.6 * prm["D_amp"] * math.sin(2 * math.pi * t_cal / PERIOD1 + prm["phase1"])
         + 0.4 * prm["D_amp"] * math.sin(2 * math.pi * t_cal / PERIOD2 + prm["phase2"]))
    return d


def true_lift(age, t_cal, prm):
    return prm["P"] + prm["A"] * math.exp(-age / prm["tau"]) + calendar_wobble(t_cal, prm)


def gen_train(t):
    prm = params(t)
    rng = random.Random(60013 + t * 977)
    rows = []
    for c in range(1, C_COHORTS + 1):
        s_c = 1 + (c - 1) * GAP
        for age in range(W):
            t_cal = s_c + age
            L = true_lift(age, t_cal, prm) + rng.gauss(0.0, prm["sigma_train"])
            rows.append((c, s_c, t_cal, age, L))
    return rows


def gen_held(t):
    """Held-out extrapolation grid: ages far beyond the visible two-week window,
    at future calendar days whose common wobble was never observed in training."""
    prm = params(t)
    rng = random.Random(9004001 + t * 613)
    pts = []
    for _ in range(N_HELD):
        age = rng.uniform(AGE_HELD_LO, AGE_HELD_HI)
        t_cal = T_VIS + rng.uniform(1.0, 400.0)
        L = true_lift(age, t_cal, prm) + rng.gauss(0.0, prm["sigma_held"])
        pts.append((age, L))
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


def eval_at(code, age):
    env = dict(ALLOWED_FUNCS)
    env["age"] = age
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
    for age, L in held:
        p = eval_at(code, age)
        if p is None:
            fail("non-finite / invalid prediction")
        d = abs(p - L) / (abs(p) + abs(L) + 1e-9)
        ds.append(min(CAP, d))
    metric = sum(ds) / len(ds)

    # baseline: constant predictor = mean of TRAIN lift column
    train = gen_train(t)
    mean_L = sum(r[4] for r in train) / len(train)
    bd = [min(CAP, abs(mean_L - L) / (abs(mean_L) + abs(L) + 1e-9)) for _, L in held]
    Bmetric = sum(bd) / len(bd)

    B = Bmetric * (1.0 + LAMBDA * 1)
    O = metric * (1.0 + LAMBDA * nodes)
    sc = min(1000.0, 100.0 * B / max(1e-12, O))
    print("metric=%.6f baseline=%.6f nodes=%d  Ratio: %.6f"
          % (metric, Bmetric, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
