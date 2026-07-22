#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the aliased fan-blade speed-law recovery task.

- Reads (t, N, fs, F_MAX) from <in>, then regenerates the hidden
  frequency-parameter law  f_true(x) = a*x^b + c  (a>0, b>1, c>=0) entirely
  from t.  The hidden law lives ONLY here and (duplicated, not imported) in
  gen.py -- never printed to the solver.
- Also regenerates a HELD-OUT extrapolation grid of drive levels well beyond
  the swept training range, plus a few adversarial corner points.  This grid
  is never shown to the solver, and -- critically -- it is scored against the
  UNALIASED true frequency: the solver's law is judged on whether it
  recovered the real physical law, not the folded reading.
- Parses the participant's closed-form frequency law -- an expression over
  the single variable `x`, numeric constants, + - * /, unary +/-, and the
  functions absv(a), minv(a,b), maxv(a,b), powv(a,b) [a must evaluate > 0].
- Scores by mean SQUARED LOG ERROR between the law's prediction and the
  (noisy) held-out truth, with a small node-count parsimony penalty:
      F = mean_k (log(pred_k) - log(true_noisy_k))^2 * (1 + LAMBDA*nodes)
      B = mean_k (log(R_geomean_train) - log(true_noisy_k))^2 * (1 + LAMBDA*1)
      Ratio = min(SCORE_CAP, 0.1 * (B/F) ** GAMMA)
  The baseline predicts the flat GEOMETRIC MEAN of the training READINGS
  (the raw aliased values, exactly what a "do nothing clever" forecast would
  use) for every held-out point -- Ratio == 0.1 when reproduced exactly.
  Because the readings are folded into [0, fs/2] while the held-out truth
  keeps growing with x, a law that actually recovers the unaliased growth
  rate drives F far below B; GAMMA compresses the resulting spread so a
  merely-plausible-shaped law does not saturate, and held-out noise plus the
  finite training sample keep even a strong recovered law below the ceiling.
"""
import sys, math, ast, random

X_LO = 1.0
F_MAX = 900.0
NOISE_SIGMA = 0.015

HELDOUT_SIGMA = 0.05
LAMBDA = 0.010
GAMMA = 0.22
SCORE_CAP = 0.90
MAX_NODES = 200
MAX_OUT_BYTES = 200000
CLAMP_LOG = 60.0

ALLOWED_FUNCS_ARITY = {"absv": 1, "minv": 2, "maxv": 2, "powv": 2}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden law (IDENTICAL to gen.py) ----------
def hidden_law(t):
    rng = random.Random(671000 + t * 7919)
    a = rng.uniform(0.8, 2.2)
    b = rng.uniform(1.25, 1.95)
    c = rng.uniform(1.0, 4.0)
    return a, b, c


def f_true(x, a, b, c):
    return a * (x ** b) + c


def x_range(t):
    return X_LO, 10.0 + 0.4 * t


_TARGET_ZMAX = {1: 0.45, 2: 0.8, 3: 1.3, 4: 2.2, 5: 3.5,
                6: 5.5, 7: 8.5, 8: 13.0, 9: 19.0, 10: 27.0}


def sampling_rate(t):
    a, b, c = hidden_law(t)
    _, X_HI = x_range(t)
    fmax_train = f_true(X_HI, a, b, c)
    target = _TARGET_ZMAX.get(t, 0.45 * (1.55 ** (t - 1)))
    rng = random.Random(671500 + t * 104729)
    jitter = rng.uniform(0.9, 1.1)
    half = fmax_train / (target * jitter)
    return 2.0 * half


def n_train(t):
    return 24 + 2 * t


def fold(freq, fs):
    half = fs / 2.0
    m = math.fmod(freq, fs)
    if m < 0.0:
        m += fs
    if m <= half:
        return m
    return fs - m


def train_rows(t):
    a, b, c = hidden_law(t)
    X_LO_, X_HI = x_range(t)
    fs = sampling_rate(t)
    half = fs / 2.0
    N = n_train(t)
    rng = random.Random(671800 + t * 13)
    log_lo, log_hi = math.log(X_LO_), math.log(X_HI)

    has_gap = t >= 4
    xs = []
    if has_gap:
        gap_lo, gap_hi = 0.42, 0.68
        n1 = int(round(N * 0.55))
        n2 = N - n1
        for i in range(n1):
            frac = (i + rng.uniform(0.05, 0.95)) / n1 * gap_lo
            xs.append(frac)
        for j in range(n2):
            frac = gap_hi + (j + rng.uniform(0.05, 0.95)) / n2 * (1.0 - gap_hi)
            xs.append(frac)
    else:
        for i in range(N):
            frac = (i + rng.uniform(0.05, 0.95)) / N
            xs.append(frac)

    rows = []
    for frac in xs:
        frac = min(0.999999, max(0.000001, frac))
        x = math.exp(log_lo + frac * (log_hi - log_lo))
        ftrue = f_true(x, a, b, c)
        f_noisy_true = ftrue * math.exp(rng.gauss(0.0, NOISE_SIGMA))
        r = fold(f_noisy_true, fs)
        r = min(half, max(0.0, r))
        rows.append((x, r))

    rng.shuffle(rows)
    return rows, fs


# held-out grid: log-spaced drive levels beyond the swept training range,
# plus a few extreme "adversarial corner" points appended at the far end.
def heldout(t):
    a, b, c = hidden_law(t)
    _, X_HI = x_range(t)
    rng = random.Random(671900 + t * 29)
    lo, hi = math.log(X_HI * 1.05), math.log(X_HI * 1.6)
    logs = [lo + k * (hi - lo) / 14.0 for k in range(15)]
    corners = [math.log(X_HI * m) for m in (1.7, 1.8, 1.9, 2.0, 2.2)]
    logs = logs + corners
    clean = []
    noisy = []
    for lg in logs:
        x = math.exp(lg)
        c_true = f_true(x, a, b, c)
        n = c_true * math.exp(rng.gauss(0.0, HELDOUT_SIGMA))
        clean.append((x, c_true))
        noisy.append((x, n))
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
            if nm != "x":
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
    for pre in ("f", "f(x)", "out", "y", "freq"):
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


_FUNCS = {
    "absv": abs,
    "minv": min,
    "maxv": max,
    "powv": _powv,
}


def eval_law(code, x):
    env = dict(_FUNCS)
    env["x"] = float(x)
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

    rows, _fs = train_rows(t)
    readings = [r for (_x, r) in rows]
    readings_pos = [max(1e-6, r) for r in readings]
    geo_log_mean = sum(math.log(r) for r in readings_pos) / len(readings_pos)

    clean, noisy = heldout(t)

    se = []
    for (x, _c), (_, ny) in zip(clean, noisy):
        pred = eval_law(code, x)
        if pred is None or pred <= 0.0:
            fail("non-finite/non-positive prediction at x=%.3g" % x)
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
