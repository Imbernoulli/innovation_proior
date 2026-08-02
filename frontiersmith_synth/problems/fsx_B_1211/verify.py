#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the ice-shelf calving-forecast task.  The solver
submits ONE closed-form expression for the calving time T as a function of
the five measured segment quantities H0, D0, c0, gamma, phi.

- Reads the test id from <in> (header), then regenerates the hidden shelf's
  critical ratio `kappa` and the HELD-OUT table of segments entirely from
  that id.  `kappa` and the RNG live ONLY here (and in gen.py, byte-for-byte
  identical).
- The held-out segments are drawn so that buttressing loss engages well
  BEFORE calving (phi sits a large, fixed margin above each segment's own
  starting ratio D0/H0), so a substantial share of the true calving horizon
  runs at the accelerated phase-2 rate -- a regime the training table (where
  phi sits just below kappa) essentially never visits.  A model fit only to
  the slow phase-1 rate observed in training extrapolates the horizon too
  long on these configurations.
- Parses the submitted expression with a strict AST whitelist:
      names     H0 D0 c0 gamma phi
      operators + - * / **  and unary +/-
      functions sqrt log exp absv minv maxv
      numeric constants
- Evaluates it on the held-out table, computes a bounded absolute log-ratio
  error per row (robust to the wide dynamic range of calving times), averages,
  and adds a small node-count parsimony penalty (minimise):
      metric = mean_i min(CAP, |ln(max(p_i,eps)) - ln(t_i)|)
      O = metric * (1 + LAMBDA * nodes)
      B = baseline_metric * (1 + LAMBDA * 1)   # baseline = const geomean(train T)
      Ratio = min(1000, 100 * B / O) / 1000
  A constant reproduces the baseline (~0.1).  A fit that only ever sees
  phase-1 (buttressing intact) dynamics has no way to identify the phase-2
  acceleration and systematically predicts calving far too late on the
  held-out table -- it stays low.  Only a model that treats D/H reaching a
  ratio (not the raw thinning/growth rate) as the trigger for a regime
  change, and correctly folds in the fixed post-trigger acceleration,
  tracks the held-out horizon.  Observation noise on both tables keeps even
  the exact law below the ceiling, leaving headroom.
"""
import sys, math, ast, random

BETA = 3.0
N_TRAIN = 60
SIGMA_TRAIN = 0.05
N_HELD = 60
SIGMA_HELD = 0.10
LAMBDA = 0.006
CAP = 1.0
EPS = 1e-6
MAX_NODES = 60
MAX_OUT_BYTES = 100000

ALLOWED_FUNCS = {
    "sqrt": lambda x: math.sqrt(x),
    "log": lambda x: math.log(x),
    "exp": lambda x: math.exp(max(-700.0, min(700.0, x))),
    "absv": abs,
    "minv": lambda a, b: a if a < b else b,
    "maxv": lambda a, b: a if a > b else b,
}
ALLOWED_NAMES = {"H0", "D0", "c0", "gamma", "phi"}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden shelf law (identical to gen.py) ----------
def shelf_kappa(t):
    rng = random.Random(500000 + t * 9176111)
    return rng.uniform(0.75, 0.92)


def true_calve_time(H0, D0, c0, gamma, phi, kappa, beta=BETA):
    t1 = (phi * H0 - D0) / (c0 + phi * gamma)
    D1 = D0 + c0 * t1
    H1 = H0 - gamma * t1
    tprime = (kappa * H1 - D1) / (c0 * (1.0 + beta) + kappa * gamma)
    return t1 + tprime


def gen_rows(t, n, rng, kappa, train, sigma):
    rows = []
    for _ in range(n):
        H0 = rng.uniform(150.0, 500.0)
        gamma = rng.uniform(2.0, 6.0)
        c0 = rng.uniform(3.0, 10.0)
        r0 = rng.uniform(0.05, 0.25)
        D0 = r0 * H0
        if train:
            phi = kappa - rng.uniform(0.01, 0.04)
        else:
            phi = r0 + rng.uniform(0.30, 0.45)
        Ttrue = true_calve_time(H0, D0, c0, gamma, phi, kappa)
        Tobs = Ttrue * math.exp(rng.gauss(0.0, sigma))
        rows.append((H0, D0, c0, gamma, phi, Tobs))
    return rows


def gen_train(t):
    kappa = shelf_kappa(t)
    rng = random.Random(111 + t * 13)
    return gen_rows(t, N_TRAIN, rng, kappa, train=True, sigma=SIGMA_TRAIN)


def gen_held(t):
    kappa = shelf_kappa(t)
    rng = random.Random(20261 + t * 15485863)
    return gen_rows(t, N_HELD, rng, kappa, train=False, sigma=SIGMA_HELD)


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
            nargs = 2 if node.func.id in ("minv", "maxv") else 1
            if node.keywords or len(node.args) != nargs:
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


def eval_at(code, H0, D0, c0, gamma, phi):
    env = dict(ALLOWED_FUNCS)
    env["H0"] = H0; env["D0"] = D0; env["c0"] = c0; env["gamma"] = gamma; env["phi"] = phi
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
    for H0, D0, c0, gamma, phi, T in held:
        p = eval_at(code, H0, D0, c0, gamma, phi)
        if p is None:
            fail("non-finite / invalid prediction")
        d = abs(math.log(max(p, EPS)) - math.log(T))
        ds.append(min(CAP, d))
    metric = sum(ds) / len(ds)

    # baseline: constant predictor = geometric mean of TRAIN T
    train = gen_train(t)
    gm = math.exp(sum(math.log(r[5]) for r in train) / len(train))
    bd = [min(CAP, abs(math.log(gm) - math.log(T))) for _, _, _, _, _, T in held]
    Bmetric = sum(bd) / len(bd)

    B = Bmetric * (1.0 + LAMBDA * 1)
    O = metric * (1.0 + LAMBDA * nodes)
    sc = min(1000.0, 100.0 * B / max(1e-12, O))
    print("metric=%.6f baseline=%.6f nodes=%d  Ratio: %.6f"
          % (metric, Bmetric, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
