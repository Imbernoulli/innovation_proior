#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the wind-tunnel drag-law recovery task (Buckingham-Pi
collapse).  The solver submits ONE closed-form expression for the drag force F
as a function of the four measured quantities rho, V, D, mu.

- Reads the case id from <in> (header), then regenerates the hidden drag law and
  the HELD-OUT EXTRAPOLATION grid (all four quantities far outside the training
  ranges -- different fluid, different scale) entirely from that id.  The law and
  its coefficients live ONLY here.
- Parses the submitted expression with a strict AST whitelist:
      names     rho V D mu
      operators + - * / **  and unary +/-
      functions sqrt log exp sig tanh absv
      numeric constants
- Evaluates it on the held-out grid, computes a bounded symmetric relative error
  per point, averages, and adds a small node-count parsimony penalty (minimise):
      metric = mean_i min(1, |p_i - t_i| / (|p_i| + |t_i|))
      O = metric * (1 + LAMBDA * nodes)
      B = baseline_metric * (1 + LAMBDA * 1)     # baseline = constant geomean(train F)
      Ratio = min(1000, 100 * B / O) / 1000
  A constant reproduces the baseline (~0.1).  A raw free-exponent power law fit
  to the training notebook cannot identify the rho / mu exponents (those columns
  are nearly flat in training) and diverges off-grid -- it stays low.  Only a
  predictor that FORCES the exponents from dimensional homogeneity
  (F = rho V^2 D^2 * g(rho V D / mu)) extrapolates; sensor noise on the held-out
  grid keeps even a good dimensional fit well below the ceiling, leaving headroom.
"""
import sys, math, ast, random

# ---- fixed design constants (mirrored byte-for-byte in gen.py) ----
NARROW   = math.log(1.03)
DSWEEP   = math.log(3.0)
VSWEEP   = math.log(8.0)
NOISE_TRAIN = 0.05
N_TRAIN  = 200

# ---- held-out / scoring constants (grader only) ----
NOISE_HELD = 0.18
N_HELD   = 120
EXTRAP_LO = math.log(30.0)     # held-out quantities are 30x..120x outside the
EXTRAP_HI = math.log(120.0)    #   training slice, in either direction
LAMBDA   = 0.004
CAP      = 1.0
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
ALLOWED_NAMES = {"rho", "V", "D", "mu"}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden drag law (identical to gen.py) ----------
def params(t):
    rng = random.Random(700000 + t * 9176111)
    a0 = rng.uniform(0.35, 1.10)
    a1 = rng.uniform(3.0, 9.0)
    a2 = rng.uniform(6.0, 22.0)
    rho0 = math.exp(rng.uniform(math.log(1.0),  math.log(1000.0)))
    V0   = math.exp(rng.uniform(math.log(5.0),  math.log(60.0)))
    D0   = math.exp(rng.uniform(math.log(0.02), math.log(3.0)))
    mu0  = math.exp(rng.uniform(math.log(1e-5), math.log(1e-2)))
    return a0, a1, a2, rho0, V0, D0, mu0


def true_F(rho, V, D, mu, a0, a1, a2):
    Re = rho * V * D / mu
    Cd = a0 + a1 * Re ** (-0.5) + a2 * Re ** (-1.0)
    return Cd * rho * V * V * D * D


def gen_train(t):
    a0, a1, a2, rho0, V0, D0, mu0 = params(t)
    rng = random.Random(111 + t * 13)
    rows = []
    for _ in range(N_TRAIN):
        V   = V0   * math.exp(rng.uniform(-VSWEEP, VSWEEP))
        D   = D0   * math.exp(rng.uniform(-DSWEEP, DSWEEP))
        rho = rho0 * math.exp(rng.uniform(-NARROW, NARROW))
        mu  = mu0  * math.exp(rng.uniform(-NARROW, NARROW))
        F   = true_F(rho, V, D, mu, a0, a1, a2) * math.exp(rng.gauss(0.0, NOISE_TRAIN))
        rows.append((rho, V, D, mu, F))
    return rows


def gen_held(t):
    """Held-out extrapolation grid: all four quantities far outside the slices."""
    a0, a1, a2, rho0, V0, D0, mu0 = params(t)
    rng = random.Random(999 + t * 7)
    pts = []
    for _ in range(N_HELD):
        def far(c):
            s = 1.0 if rng.random() < 0.5 else -1.0
            return c * math.exp(s * rng.uniform(EXTRAP_LO, EXTRAP_HI))
        rho = far(rho0); V = far(V0); D = far(D0); mu = far(mu0)
        F = true_F(rho, V, D, mu, a0, a1, a2) * math.exp(rng.gauss(0.0, NOISE_HELD))
        pts.append((rho, V, D, mu, F))
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


def eval_at(code, rho, V, D, mu):
    env = dict(ALLOWED_FUNCS)
    env["rho"] = rho; env["V"] = V; env["D"] = D; env["mu"] = mu
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
    for rho, V, D, mu, F in held:
        p = eval_at(code, rho, V, D, mu)
        if p is None:
            fail("non-finite / invalid prediction")
        d = abs(p - F) / (abs(p) + abs(F) + 1e-30)
        ds.append(min(CAP, d))
    metric = sum(ds) / len(ds)

    # baseline: constant predictor = geometric mean of TRAIN F
    train = gen_train(t)
    gm = math.exp(sum(math.log(r[4]) for r in train) / len(train))
    bd = [min(CAP, abs(gm - F) / (abs(gm) + abs(F) + 1e-30)) for _, _, _, _, F in held]
    Bmetric = sum(bd) / len(bd)

    B = Bmetric * (1.0 + LAMBDA * 1)
    O = metric * (1.0 + LAMBDA * nodes)
    sc = min(1000.0, 100.0 * B / max(1e-12, O))
    print("metric=%.6f baseline=%.6f nodes=%d  Ratio: %.6f"
          % (metric, Bmetric, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
