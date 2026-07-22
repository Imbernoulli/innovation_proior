#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the dual-Arrhenius-crossover dough-rise task.

- Reads the test id from <in>, then regenerates the hidden two-channel
  crossover law
      k1(T) = A1*exp(-theta1/T)     (fermentation channel, rises with T)
      k2(T) = A2*exp(+theta2/T)     (stability channel, falls with T)
      r(T)  = k1(T)*k2(T) / (k1(T)+k2(T))     (harmonic combination)
  entirely from that id. The hidden law lives ONLY here (and, separately,
  inside gen.py -- never printed to the solver).
- Also regenerates a HELD-OUT grid: temperatures both well BELOW the
  proofing window (fridge-retard regime, fermentation-channel-limited) and
  well ABOVE it (oven-overshoot regime, where the stability channel takes
  over and the rise rate turns over and stalls). This grid is never shown
  to the solver.
- Parses the participant's closed-form rate law -- an expression over the
  single variable `T`, numeric constants, + - * /, unary +/-, and the
  functions absv(a), minv(a,b), maxv(a,b), powv(a,b) [a must evaluate > 0],
  expv(a) [= e^a, any finite a].
- Scores by mean SQUARED LOG ERROR between the law's prediction and the
  (noisy) held-out truth, log(pred) vs log(true), with a small node-count
  parsimony penalty:
      F = mean_k (log(pred_k) - log(true_noisy_k))^2 * (1 + LAMBDA*nodes)
      B = mean_k (log(r_geomean_train) - log(true_noisy_k))^2 * (1 + LAMBDA*1)
      Ratio = min(SCORE_CAP, 0.1 * (B/F) ** GAMMA)
  The baseline predicts the flat GEOMETRIC MEAN of the training r values for
  every held-out point (Ratio == 0.1 when reproduced exactly, i.e. B/F == 1).
  Squared LOG error is used because it rewards getting the DECAY/TURNOVER
  RATE right, not just the absolute scale. GAMMA < 1 compresses B/F so a
  merely-correct-shaped law does not saturate. Noise on the held-out grid
  plus the finite training sample keep even a strong recovered law below
  the SCORE_CAP -- there is room to improve. Report the highest Ratio you can.
"""
import sys, math, ast, random

T_LO = 298.0
T_HI = 315.0
T_REF = (T_LO + T_HI) / 2.0
N_TRAIN = 120
NOISE_SIGMA = 0.012

HELDOUT_SIGMA = 0.07          # held-out observation-noise floor (irreducible)
LAMBDA = 0.008
GAMMA = 0.32                  # sub-linear compression exponent on B/F (headroom)
SCORE_CAP = 0.90              # hard ceiling: never saturate to 1.0
MAX_NODES = 200
MAX_OUT_BYTES = 200000
CLAMP_LOG = 60.0
EXPV_ARG_CAP = 80.0           # guard against overflow in expv(a)

ALLOWED_FUNCS_ARITY = {"absv": 1, "minv": 2, "maxv": 2, "powv": 2, "expv": 1}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden law (IDENTICAL to gen.py) ----------
def hidden_law(t):
    rng = random.Random(925000 + t * 7919)
    theta1 = rng.uniform(2000.0, 2800.0)
    theta2 = rng.uniform(900.0, 1500.0)
    k1_ref = rng.uniform(400.0, 900.0)
    k2_ref = rng.uniform(1500.0, 3000.0)
    A1 = k1_ref * math.exp(theta1 / T_REF)
    A2 = k2_ref * math.exp(-theta2 / T_REF)
    return A1, theta1, A2, theta2


def rate_true(T, A1, theta1, A2, theta2):
    k1 = A1 * math.exp(-theta1 / T)
    k2 = A2 * math.exp(theta2 / T)
    return k1 * k2 / (k1 + k2)


def train_rows(t):
    A1, theta1, A2, theta2 = hidden_law(t)
    rng = random.Random(11000 + t * 13)
    rows = []
    for i in range(N_TRAIN):
        frac = (i + rng.uniform(0.05, 0.95)) / N_TRAIN
        frac = min(0.999999, max(0.000001, frac))
        T = T_LO + frac * (T_HI - T_LO)
        clean = rate_true(T, A1, theta1, A2, theta2)
        noisy = clean * math.exp(rng.gauss(0.0, NOISE_SIGMA))
        rows.append((T, noisy))
    rows.sort(key=lambda r: r[0])
    return rows


# held-out grid: temperatures below the proofing window (fridge retard) and
# well above it (oven overshoot, where the stability channel takes over).
_HELDOUT_LOW = [270.0, 274.0, 278.0, 282.0, 286.0, 290.0, 293.0, 296.0]
_HELDOUT_HIGH = [318.0, 324.0, 330.0, 336.0, 342.0, 348.0, 354.0, 360.0,
                 368.0, 376.0, 386.0, 398.0]


def heldout(t):
    A1, theta1, A2, theta2 = hidden_law(t)
    rng = random.Random(4400 + t * 29)
    Ts = _HELDOUT_LOW + _HELDOUT_HIGH
    clean = []
    noisy = []
    for T in Ts:
        c = rate_true(T, A1, theta1, A2, theta2)
        n = c * math.exp(rng.gauss(0.0, HELDOUT_SIGMA))
        clean.append((T, c))
        noisy.append((T, n))
    return clean, noisy


# ---------- expression parsing / validation ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
)


def _validate_ast(tree):
    used = set()
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return None, "disallowed syntax %s" % type(node).__name__
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS_ARITY):
                return None, "disallowed call"
            if node.keywords:
                return None, "keyword args not allowed"
            need = ALLOWED_FUNCS_ARITY[node.func.id]
            if len(node.args) != need:
                return None, "%s takes %d arg(s)" % (node.func.id, need)
        if isinstance(node, ast.Name):
            nm = node.id
            if nm in ALLOWED_FUNCS_ARITY:
                continue
            if nm != "T":
                return None, "unknown name %s" % nm
            used.add(nm)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                return None, "non-numeric constant"
            v = float(node.value)
            if v != v or v in (float("inf"), float("-inf")):
                return None, "non-finite constant"
    return used, None


def _count_nodes(tree):
    return sum(1 for nd in ast.walk(tree)
               if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant)))


def parse_law(raw):
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        fail("empty output")
    text = lines[-1]
    low = text.lower()
    for pre in ("r", "r(t)", "out", "y", "f(t)"):
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
    used, err = _validate_ast(tree)
    if err:
        fail(err)
    nodes = _count_nodes(tree)
    if nodes > MAX_NODES:
        fail("law too large (%d nodes)" % nodes)
    try:
        code = compile(tree, "<law>", "eval")
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


def _expv(a):
    if not isinstance(a, (int, float)) or isinstance(a, bool):
        return float("nan")
    if a != a or a in (float("inf"), float("-inf")):
        return float("nan")
    if a > EXPV_ARG_CAP or a < -EXPV_ARG_CAP:
        return float("nan")   # would overflow/underflow -- treat as infeasible
    try:
        return math.exp(a)
    except Exception:
        return float("nan")


_FUNCS = {
    "absv": abs,
    "minv": min,
    "maxv": max,
    "powv": _powv,
    "expv": _expv,
}


def eval_law(code, T):
    env = dict(_FUNCS)
    env["T"] = float(T)
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

    code, nodes = parse_law(text)

    rows = train_rows(t)
    rs = [r for (_, r) in rows]
    geo_log_mean = sum(math.log(r) for r in rs) / len(rs)

    clean, noisy = heldout(t)

    se = []
    for (T, _c), (_, ny) in zip(clean, noisy):
        pred = eval_law(code, T)
        if pred is None or pred <= 0.0:
            fail("non-finite/non-positive prediction at T=%.3g" % T)
        lp = max(-CLAMP_LOG, min(CLAMP_LOG, math.log(pred)))
        lt = math.log(ny)
        se.append((lp - lt) ** 2)
    F_mse = sum(se) / len(se)

    se_b = [(geo_log_mean - math.log(ny)) ** 2 for (_, ny) in noisy]
    B_mse = sum(se_b) / len(se_b)

    F = F_mse * (1.0 + LAMBDA * nodes)
    B = B_mse * (1.0 + LAMBDA * 1)
    ratio_raw = B / max(1e-9, F)
    sc = min(SCORE_CAP, 0.1 * (ratio_raw ** GAMMA))
    print("heldout_MSLE=%.6f baseline_MSLE=%.6f nodes=%d B/F=%.4f  Ratio: %.6f"
          % (F_mse, B_mse, nodes, ratio_raw, sc))


if __name__ == "__main__":
    main()
