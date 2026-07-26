#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the dielectric-breakdown-crossover task.

- Reads the test id from <in>, then regenerates the hidden two-channel
  breakdown law entirely from that id (IDENTICAL to gen.py -- the hidden
  law lives ONLY here and in gen.py, never printed to the solver):

      V1(d,T) = V1ref*(d/D_REF)^p1*(T/T_REF)^q1      (avalanche-like)
      V2(d,T) = V2ref*(d/D_REF)^p2*(T/T_REF)^q2      (tunneling-like)
      V(d,T)  = softmin_k(V1,V2)                     (weakest-link, smoothed)

- Regenerates a HELD-OUT grid: 20 points clustered around the four EXTREME
  CORNERS of the (d,T) square, well outside the [40,90]x[280,340] training
  window (the "stress corners": thin+cold, thin+hot, thick+cold,
  thick+hot). This grid is never shown to the solver.
- Parses the participant's closed-form voltage law -- an expression over
  the two variables `d`, `T`, numeric constants, + - * /, unary +/-, and
  the functions absv(a), minv(a,b), maxv(a,b), powv(a,b) [a must evaluate
  positive], expv(a) [=e^a, any finite a], logv(a) [=ln(a), a must
  evaluate positive].
- Scores by mean SQUARED LOG ERROR between the law's prediction and the
  (noisy) held-out truth at the stress corners, with a small node-count
  parsimony penalty:
      F = mean_k (log(pred_k) - log(true_noisy_k))^2 * (1 + LAMBDA*nodes)
      B = mean_k (log(Vbar)   - log(true_noisy_k))^2 * (1 + LAMBDA*1)
      Ratio = min(SCORE_CAP, 0.1 * (B/F) ** GAMMA)
  Vbar is the flat GEOMETRIC MEAN of the training V values (the checker's
  own trivial baseline construction): Ratio == 0.1 exactly when that flat
  baseline is reproduced (B/F == 1). Squared LOG error rewards getting the
  SATURATED, channel-specific asymptote right at each corner, not just the
  overall scale. GAMMA<1 compresses B/F so a merely-plausible-shaped law
  does not saturate; SCORE_CAP<1 is a hard ceiling. Measurement noise on
  both the training rows and the held-out grid keeps even a strong,
  correctly-structured law below the ceiling.
"""
import sys
import math
import ast
import random

D_LO, D_HI = 40.0, 90.0
T_LO, T_HI = 280.0, 340.0
D_REF = 65.0
T_REF = 310.0
N_TRAIN = 140
NOISE_SIGMA = 0.020

SEED_BASE = 114100
SEED_MULT = 7919
ROW_SEED_BASE = 22100
ROW_SEED_MULT = 131
HELDOUT_SEED_BASE = 55700
HELDOUT_SEED_MULT = 383

HELDOUT_SIGMA = 0.05           # held-out observation-noise floor (irreducible)
LAMBDA = 0.006
GAMMA = 0.34                    # sub-linear compression exponent on B/F (headroom)
SCORE_CAP = 0.90                 # hard ceiling: never saturate to 1.0
MAX_NODES = 260
MAX_OUT_BYTES = 200000
CLAMP_LOG = 60.0
EXPV_ARG_CAP = 80.0

ALLOWED_FUNCS_ARITY = {"absv": 1, "minv": 2, "maxv": 2, "powv": 2, "expv": 1, "logv": 1}

# 20 held-out points clustered around the four extreme corners of the
# (d,T) square, well outside the [40,90]x[280,340] training window.
_HELDOUT_POINTS = [
    # thin + cold (both channels' d- and T-signals agree: avalanche wins)
    (15.0, 170.0), (15.0, 200.0), (20.0, 180.0), (25.0, 170.0), (17.0, 190.0),
    # thin + hot (d says avalanche, T says tunneling -- MIXED, instance-dependent)
    (15.0, 520.0), (15.0, 480.0), (20.0, 500.0), (25.0, 520.0), (17.0, 490.0),
    # thick + cold (d says tunneling, T says avalanche -- MIXED, instance-dependent)
    (165.0, 170.0), (165.0, 200.0), (140.0, 180.0), (165.0, 190.0), (150.0, 170.0),
    # thick + hot (both channels' d- and T-signals agree: tunneling wins)
    (165.0, 520.0), (165.0, 480.0), (140.0, 500.0), (165.0, 490.0), (150.0, 520.0),
]


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden law (IDENTICAL to gen.py) ----------
def hidden_law(t):
    rng = random.Random(SEED_BASE + t * SEED_MULT)
    p1 = rng.uniform(1.00, 1.20)
    q1 = rng.uniform(0.55, 0.85)
    p2 = rng.uniform(0.45, 0.65)
    q2 = rng.uniform(-0.75, -0.45)
    V1ref = rng.uniform(63.0, 80.0)
    V2ref = rng.uniform(63.0, 80.0)
    k = rng.uniform(0.05, 0.09)
    return p1, q1, p2, q2, V1ref, V2ref, k


def branch1(d, T, p1, q1, V1ref):
    return V1ref * (d / D_REF) ** p1 * (T / T_REF) ** q1


def branch2(d, T, p2, q2, V2ref):
    return V2ref * (d / D_REF) ** p2 * (T / T_REF) ** q2


def v_true(d, T, params):
    p1, q1, p2, q2, V1ref, V2ref, k = params
    v1 = branch1(d, T, p1, q1, V1ref)
    v2 = branch2(d, T, p2, q2, V2ref)
    m = min(v1, v2)
    z = math.exp(-k * (v1 - m)) + math.exp(-k * (v2 - m))
    return m - math.log(z) / k


def train_rows(t):
    params = hidden_law(t)
    rng = random.Random(ROW_SEED_BASE + t * ROW_SEED_MULT)
    rows = []
    side = int(round(N_TRAIN ** 0.5))
    while side * side < N_TRAIN:
        side += 1
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N_TRAIN:
                break
            fd = (i + rng.uniform(0.08, 0.92)) / side
            fT = (j + rng.uniform(0.08, 0.92)) / side
            fd = min(0.999999, max(0.000001, fd))
            fT = min(0.999999, max(0.000001, fT))
            d = D_LO + fd * (D_HI - D_LO)
            T = T_LO + fT * (T_HI - T_LO)
            clean = v_true(d, T, params)
            noisy = clean * math.exp(rng.gauss(0.0, NOISE_SIGMA))
            rows.append((d, T, noisy))
            idx += 1
    rng.shuffle(rows)
    rows = rows[:N_TRAIN]
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def heldout(t):
    params = hidden_law(t)
    rng = random.Random(HELDOUT_SEED_BASE + t * HELDOUT_SEED_MULT)
    clean = []
    noisy = []
    for (d, T) in _HELDOUT_POINTS:
        c = v_true(d, T, params)
        n = c * math.exp(rng.gauss(0.0, HELDOUT_SIGMA))
        clean.append((d, T, c))
        noisy.append((d, T, n))
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
            if nm not in ("d", "T"):
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
    for pre in ("v", "v(d,t)", "out", "y"):
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
        return float("nan")
    try:
        return math.exp(a)
    except Exception:
        return float("nan")


def _logv(a):
    if not isinstance(a, (int, float)) or isinstance(a, bool):
        return float("nan")
    if a <= 0.0:
        return float("nan")
    try:
        return math.log(a)
    except Exception:
        return float("nan")


def _minv(a, b):
    if not isinstance(a, (int, float)) or isinstance(a, bool):
        return float("nan")
    if not isinstance(b, (int, float)) or isinstance(b, bool):
        return float("nan")
    return min(a, b)


def _maxv(a, b):
    if not isinstance(a, (int, float)) or isinstance(a, bool):
        return float("nan")
    if not isinstance(b, (int, float)) or isinstance(b, bool):
        return float("nan")
    return max(a, b)


def _absv(a):
    if not isinstance(a, (int, float)) or isinstance(a, bool):
        return float("nan")
    return abs(a)


_FUNCS = {
    "absv": _absv,
    "minv": _minv,
    "maxv": _maxv,
    "powv": _powv,
    "expv": _expv,
    "logv": _logv,
}


def eval_law(code, d, T):
    env = dict(_FUNCS)
    env["d"] = float(d)
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
    vs = [v for (_, _, v) in rows]
    geo_log_mean = sum(math.log(v) for v in vs) / len(vs)

    clean, noisy = heldout(t)

    se = []
    for (d, T, _c), (_, _, ny) in zip(clean, noisy):
        pred = eval_law(code, d, T)
        if pred is None or pred <= 0.0:
            fail("non-finite/non-positive prediction at d=%.3g,T=%.3g" % (d, T))
        lp = max(-CLAMP_LOG, min(CLAMP_LOG, math.log(pred)))
        lt = math.log(ny)
        se.append((lp - lt) ** 2)
    F_mse = sum(se) / len(se)

    se_b = [(geo_log_mean - math.log(ny)) ** 2 for (_, _, ny) in noisy]
    B_mse = sum(se_b) / len(se_b)

    F = F_mse * (1.0 + LAMBDA * nodes)
    B = B_mse * (1.0 + LAMBDA * 1)
    ratio_raw = B / max(1e-9, F)
    sc = min(SCORE_CAP, 0.1 * (ratio_raw ** GAMMA))
    print("heldout_MSLE=%.6f baseline_MSLE=%.6f nodes=%d B/F=%.4f  Ratio: %.6f"
          % (F_mse, B_mse, nodes, ratio_raw, sc))


if __name__ == "__main__":
    main()
