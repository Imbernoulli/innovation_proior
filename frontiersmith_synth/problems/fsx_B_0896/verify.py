#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the hopper-discharge law-recovery task ("grain silo
master's secret gauge"). The solver submits ONE closed-form expression for
the discharge rate Q as a function of the aperture width D, grain diameter
d, and grain bulk density rho.

- Reads the test id from <in> (header), then regenerates the hidden
  discharge law
      Q(D, d, rho) = C * rho * (D - k*d) ^ p          (D > k*d)
  and the HELD-OUT LARGE-APERTURE / UNSEEN-GRAIN-SIZE grid entirely from
  that id (routine IDENTICAL to gen.py -- lives ONLY here and in gen.py,
  never printed to the solver).
- Parses the submitted expression with a strict AST whitelist:
      names     D d rho
      operators + - * /  and unary +/-
      functions absv(a)            -- |a|
                powv(a, b)         -- a**b, requires a > 0 (else rejected)
      numeric constants
  (powv instead of ** avoids complex results from a negative base raised to
  a fractional power -- a submission that hits that case is rejected, not
  silently coerced.)
- Evaluates it on the held-out grid, scores by mean SQUARED LOG ERROR
  (rewards recovering the correct discharge LAW -- offset and exponent
  together -- not just matching one scale), with a small node-count
  parsimony penalty, against the flat geometric-mean-of-training baseline:

      F  = mean_k (log(pred_k) - log(true_noisy_k))^2 * (1 + LAMBDA*nodes)
      B  = mean_k (log(qbar_train) - log(true_noisy_k))^2 * (1 + LAMBDA*1)
      Ratio = min(SCORE_CAP, 0.1 * (B/F) ** GAMMA)

  Reproducing the flat training geometric mean everywhere gives B/F == 1,
  i.e. Ratio == 0.1. A law that never resolves the (D - k*d) offset or its
  exponent p tracks the training band's scale reasonably (it saturates
  B/F only modestly) but is increasingly wrong on the held-out grid, where
  apertures are 2-20x wider and grain sizes are entirely unseen. Held-out
  sensor noise and the sub-linear GAMMA compression keep even a correctly
  shaped law well below the hard cap -- there is room above the reference
  solutions. Report the highest Ratio you can.
"""
import sys, math, ast, random

# ---- fixed design constants (mirrored byte-for-byte in gen.py) ----
SEED_BASE = 896000
D_TR_LO, D_TR_HI = 6.0, 20.0
d_TR_LO, d_TR_HI = 0.2, 1.0
RHO_LO, RHO_HI = 1.0, 3.0
N_TRAIN = 200
NOISE_TRAIN = 0.02

# ---- held-out / scoring constants (grader only) ----
D_HE_LO, D_HE_HI = 40.0, 120.0
d_HE_LO, d_HE_HI = 1.2, 3.2
N_HELD = 140
NOISE_HELD = 0.12
LAMBDA = 0.010
GAMMA = 0.22
SCORE_CAP = 0.90
MAX_NODES = 80
MAX_OUT_BYTES = 100000
CLAMP_LOG = 80.0

ALLOWED_FUNCS_ARITY = {"absv": 1, "powv": 2}
ALLOWED_NAMES = {"D", "d", "rho"}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden discharge law (IDENTICAL to gen.py) ----------
def params(t):
    rng = random.Random(SEED_BASE + t * 104729)
    p_exp = rng.uniform(2.2, 2.9)
    k_off = rng.uniform(1.2, 2.4)
    C = rng.uniform(0.4, 1.6)
    return p_exp, k_off, C


def true_Q(D, d, rho, p_exp, k_off, C):
    X = D - k_off * d
    if X <= 0.0:
        return None
    return C * rho * (X ** p_exp)


def gen_train(t):
    p_exp, k_off, C = params(t)
    rng = random.Random(11000 + t * 131)
    rows = []
    while len(rows) < N_TRAIN:
        D = rng.uniform(D_TR_LO, D_TR_HI)
        d = rng.uniform(d_TR_LO, d_TR_HI)
        rho = rng.uniform(RHO_LO, RHO_HI)
        q = true_Q(D, d, rho, p_exp, k_off, C)
        if q is None:
            continue
        q *= math.exp(rng.gauss(0.0, NOISE_TRAIN))
        rows.append((D, d, rho, q))
    return rows


def gen_held(t):
    """Held-out grid: LARGE apertures + grain sizes never seen in training."""
    p_exp, k_off, C = params(t)
    rng = random.Random(87000 + t * 733)
    pts = []
    while len(pts) < N_HELD:
        D = rng.uniform(D_HE_LO, D_HE_HI)
        d = rng.uniform(d_HE_LO, d_HE_HI)
        rho = rng.uniform(RHO_LO, RHO_HI)
        q = true_Q(D, d, rho, p_exp, k_off, C)
        if q is None:
            continue
        q *= math.exp(rng.gauss(0.0, NOISE_HELD))
        pts.append((D, d, rho, q))
    return pts


# ---------- expression parsing / validation ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
)


def _validate_ast(tree):
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return "disallowed syntax %s" % type(node).__name__
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS_ARITY):
                return "disallowed call"
            if node.keywords:
                return "keyword args not allowed"
            need = ALLOWED_FUNCS_ARITY[node.func.id]
            if len(node.args) != need:
                return "%s takes %d arg(s)" % (node.func.id, need)
        if isinstance(node, ast.Name):
            nm = node.id
            if nm in ALLOWED_FUNCS_ARITY:
                continue
            if nm not in ALLOWED_NAMES:
                return "unknown name %s" % nm
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
        fail("empty output")
    text = lines[-1]
    low = text.lower()
    for pre in ("q", "q(d,d,rho)", "out"):
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
    err = _validate_ast(tree)
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


def eval_expr(code, D, d, rho):
    env = dict(_FUNCS)
    env["D"] = float(D); env["d"] = float(d); env["rho"] = float(rho)
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

    code, nodes = parse_expr(text)

    train = gen_train(t)
    ys = [r[3] for r in train]
    geo_log_mean = sum(math.log(y) for y in ys) / len(ys)

    held = gen_held(t)

    se = []
    for D, d, rho, y_noisy in held:
        pred = eval_expr(code, D, d, rho)
        if pred is None or pred <= 0.0:
            fail("non-finite/non-positive prediction at D=%.4g d=%.4g rho=%.4g" % (D, d, rho))
        lp = max(-CLAMP_LOG, min(CLAMP_LOG, math.log(pred)))
        lt = math.log(y_noisy)
        se.append((lp - lt) ** 2)
    F_msle = sum(se) / len(se)

    se_b = [(geo_log_mean - math.log(y_noisy)) ** 2 for (_, _, _, y_noisy) in held]
    B_msle = sum(se_b) / len(se_b)

    F = F_msle * (1.0 + LAMBDA * nodes)
    B = B_msle * (1.0 + LAMBDA * 1)
    ratio_raw = B / max(1e-9, F)
    sc = min(SCORE_CAP, 0.1 * (ratio_raw ** GAMMA))
    print("heldout_MSLE=%.6f baseline_MSLE=%.6f nodes=%d B/F=%.4f  Ratio: %.6f"
          % (F_msle, B_msle, nodes, ratio_raw, sc))


if __name__ == "__main__":
    main()
