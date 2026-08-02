#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the delivery-lead-time feedback recovery task.

The solver submits ONE closed-form expression for the lead time L as a function
of O (orders placed this period), and optionally D (raw demand estimate), Cap
(declared capacity) and L0 (declared free-flow lead time) -- all read straight
from the training header/rows.

- Reads (n, t, Cap, L0) and the n training rows (D, O, L) from <in> (exactly what
  gen.py printed).  Uses them only for: (a) meanL, the constant-predictor
  baseline; (b) the LAST observed L, to seed the held-out continuation.
- Regenerates the hidden per-instance parameters (g, D0, sigma, h) from testId via
  `params(t)` (byte-identical to gen.py) and simulates a HELD-OUT DEMAND-SHOCK
  window entirely inside this file -- a sustained higher demand level that,
  through the SAME order-inflation-feedback loop, settles at a materially higher
  stable order/utilization level than anything seen in training (a short burn-in
  is discarded so the recorded points are past the transient).  The ground truth
  lives ONLY here; it is never written to <in>.
- Parses the submitted expression with a strict AST whitelist (names O D Cap L0;
  operators + - * / ** and unary +/-; functions sqrt log exp sig tanh absv;
  numeric constants), evaluates it at each held-out row's O (with D, Cap, L0
  also bound), and scores from a bounded log-ratio error + a node-count
  parsimony penalty (maximise):
      err_i   = |ln(pred_i / true_i)|              (pred<=0 -> big fixed error)
      quality = mean_i 1/(1 + Q_K * err_i)
      F = quality / (1 + LAMBDA*nodes)
      B = quality_of_constant(meanL) / (1 + LAMBDA*1)
      Ratio = min(1000, 100*F/B) / 1000
  A constant reproduces the baseline (~0.1).  A plain OLS line fit to the (O, L)
  training cloud (the obvious "queueing looks linear here" approach) captures the
  LOCAL slope fine but has no pole -- it systematically underpredicts once the
  held-out shock pushes utilization well above anything trained on.  Recovering
  the true L0 + g*O/(Cap-O) shape (e.g. via the reciprocal linearization
  1/(L-L0) = (Cap/g)*(1/O) - 1/g, which is EXACTLY linear even though L(O) barely
  curves within the narrow stable-regime O range) extrapolates correctly.  Noise
  keeps even the correct shape well below the ceiling.
"""
import sys, math, ast, random

# ---- fixed design constants (mirrored byte-for-byte in gen.py for params/n_train) ----
SEED_BASE = 20260726
SEED_MULT = 104729

N_HELD = 30
BURN_IN = 40
LAMBDA = 0.01
Q_K = 6.0
MAX_NODES = 40
MAX_OUT_BYTES = 100000
MAX_N = 100000

ALLOWED_FUNCS = {
    "sqrt": lambda x: math.sqrt(x),
    "log":  lambda x: math.log(x),
    "exp":  lambda x: math.exp(max(-700.0, min(700.0, x))),
    "sig":  lambda x: 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x)))),
    "tanh": math.tanh,
    "absv": abs,
}
ALLOWED_NAMES = {"O", "D", "Cap", "L0"}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden per-instance parameters (identical to gen.py) ----------
def params(t):
    rng = random.Random(SEED_BASE + t * SEED_MULT)
    Cap = rng.uniform(90.0, 170.0)
    L0 = rng.uniform(1.8, 3.4)
    g = rng.uniform(1.5, 3.5)
    util0 = rng.uniform(0.20, 0.30)
    D0 = util0 * Cap
    sigma = D0 * rng.uniform(0.05, 0.08)
    h = rng.uniform(0.008, 0.018)
    return Cap, L0, g, D0, sigma, h


def target_rho(t):
    return 0.40 + 0.34 * (t - 1) / 9.0   # 0.40 .. 0.74 (safely below the stable-branch limit)


def solve_Dshock(Cap, h, g, u):
    """Demand level whose STABLE order-inflation-feedback fixed point sits at
    utilization u = O*/Cap. (From O=D(1+h*g*O/(Cap-O)) solved for D at a target O.)"""
    K = h * g
    d = u * (1.0 - u) / (1.0 - u * (1.0 - K))
    return d * Cap


def gen_held(t, L_prev_seed):
    """Held-out demand-shock window: same feedback+queueing law, sustained higher
    demand, burn-in discarded so the recorded points sit near the new (higher,
    still stable) utilization level -- a regime never seen in training."""
    Cap, L0, g, D0, sigma, h = params(t)
    rng = random.Random(50000 + t * 6113)
    rho = target_rho(t)
    Dshock = solve_Dshock(Cap, h, g, rho)
    L_prev = L_prev_seed
    for _ in range(BURN_IN):
        Dt = max(1e-3, Dshock + rng.gauss(0.0, sigma * 1.2))
        Ot = max(1e-3, Dt * (1.0 + h * (L_prev - L0)))
        Ot = min(Ot, 0.92 * Cap)
        Lt_true = L0 + g * Ot / (Cap - Ot)
        L_prev = Lt_true * (1.0 + rng.gauss(0.0, 0.05))
    rows = []
    for _ in range(N_HELD):
        Dt = max(1e-3, Dshock + rng.gauss(0.0, sigma * 1.2))
        Ot = max(1e-3, Dt * (1.0 + h * (L_prev - L0)))
        Ot = min(Ot, 0.92 * Cap)
        Lt_true = L0 + g * Ot / (Cap - Ot)
        Lt = Lt_true * (1.0 + rng.gauss(0.0, 0.05))
        L_prev = Lt
        rows.append((Dt, Ot, Lt))
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


def eval_at(code, O, D, Cap, L0):
    env = dict(ALLOWED_FUNCS)
    env["O"] = O; env["D"] = D; env["Cap"] = Cap; env["L0"] = L0
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


def quality(pred, true_):
    if pred is None:
        return None
    if pred <= 0.0:
        r = 50.0
    else:
        r = abs(math.log(pred / true_))
    return 1.0 / (1.0 + Q_K * r)


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            lines = fh.read().split("\n")
        header = lines[0].split()
        n = int(header[0]); t = int(header[1])
        Cap = float(header[2]); L0 = float(header[3])
        if n < 1 or n > MAX_N or t < 1 or t > 100000:
            raise ValueError("bad header")
        train = []
        for i in range(1, n + 1):
            parts = lines[i].split()
            Dv, Ov, Lv = float(parts[0]), float(parts[1]), float(parts[2])
            train.append((Dv, Ov, Lv))
    except Exception:
        fail("bad instance file")

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace")

    code, nodes = parse_expr(text)

    meanL = sum(r[2] for r in train) / len(train)
    L_prev_seed = train[-1][2]
    held = gen_held(t, L_prev_seed)

    qs = []
    for D, O, L in held:
        p = eval_at(code, O, D, Cap, L0)
        q = quality(p, L)
        if q is None:
            fail("non-finite / invalid prediction")
        qs.append(q)
    Fraw = sum(qs) / len(qs)
    F = Fraw / (1.0 + LAMBDA * nodes)

    qb = [quality(meanL, L) for _, _, L in held]
    Braw = sum(qb) / len(qb)
    B = Braw / (1.0 + LAMBDA * 1)

    sc = min(1000.0, 100.0 * F / max(1e-12, B))
    print("quality=%.6f baseline=%.6f nodes=%d  Ratio: %.6f"
          % (Fraw, Braw, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
