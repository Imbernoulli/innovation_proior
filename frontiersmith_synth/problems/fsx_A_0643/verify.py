#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the drag-law crossover-recovery task.

- Reads the test id from <in>, then regenerates the hidden two-term crossover
  law  Cd(Re) = A*Re^p + B*Re^q  (p < q < 0)  entirely from that id.  The
  hidden law lives ONLY here (and, separately, inside gen.py -- never printed
  to the solver).
- Also regenerates a HELD-OUT extrapolation grid: Reynolds numbers spanning
  several decades BEYOND the flume's reachable training range, including a
  handful of extreme "adversarial corner" points.  This grid is never shown
  to the solver.
- Parses the participant's closed-form drag-coefficient law -- an expression
  over the single variable `Re`, constants, + - * /, unary +/-, and the
  functions absv(a), minv(a,b), maxv(a,b), powv(a,b) [a must evaluate > 0].
- Scores by mean SQUARED LOG ERROR between the law's prediction and the
  (noisy) held-out truth, log(pred) vs log(true), with a small node-count
  parsimony penalty:
      F = mean_k (log(pred_k) - log(true_noisy_k))^2 * (1 + LAMBDA*nodes)
      B = mean_k (log(Cd_geomean_train) - log(true_noisy_k))^2 * (1 + LAMBDA*1)
      Ratio = min(1.0, 0.1 * (B/F) ** GAMMA)
  The baseline predicts the flat GEOMETRIC MEAN of the training Cd values for
  every held-out point (Ratio == 0.1 when reproduced exactly, i.e. B/F == 1).
  Squared LOG error is used because Cd spans several decades between the
  training range and the extrapolation grid -- it rewards getting the DECAY
  RATE right, not just the absolute magnitude at one scale. The B/F ratio
  itself can span orders of magnitude once the decay rate is nearly right
  (the extrapolation grid is many decades out), so a sub-linear power GAMMA
  compresses it back into [0,1] without letting a merely-correct-shaped law
  saturate the score. Noise on the held-out grid plus the finite training
  sample keep even a strong recovered law well below the ceiling: there is
  room to improve. Report the highest Ratio you can.
"""
import sys, math, ast, random, re

RE_LO = 0.5
RE_HI = 25.0
N_TRAIN = 90
NOISE_SIGMA = 0.020

HELDOUT_SIGMA = 0.08          # held-out observation-noise floor (irreducible)
LAMBDA = 0.010
GAMMA = 0.35                  # sub-linear compression exponent on B/F (headroom)
SCORE_CAP = 0.90               # hard ceiling: never saturate to 1.0
MAX_NODES = 200
MAX_OUT_BYTES = 200000
CLAMP_LOG = 60.0              # clamp |log value| to keep log-error finite-ish

ALLOWED_FUNCS_ARITY = {"absv": 1, "minv": 2, "maxv": 2, "powv": 2}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden law (IDENTICAL to gen.py) ----------
def hidden_law(t):
    rng = random.Random(643000 + t * 7919)
    p = -1.0 + rng.uniform(-0.08, 0.08)
    A = rng.uniform(18.0, 30.0)
    plan_q = {1: -0.12, 2: -0.18, 3: -0.35, 4: -0.40, 5: -0.15,
              6: -0.10, 7: -0.45, 8: -0.20, 9: -0.30, 10: -0.08}
    q = plan_q.get(t, -0.20) + rng.uniform(-0.02, 0.02)
    B = rng.uniform(0.15, 0.35)
    return p, A, q, B


def Cd_true(Re, p, A, q, B):
    return A * (Re ** p) + B * (Re ** q)


def train_rows(t):
    p, A, q, B = hidden_law(t)
    rng = random.Random(90210 + t * 13)
    log_lo, log_hi = math.log(RE_LO), math.log(RE_HI)
    rows = []
    for i in range(N_TRAIN):
        frac = (i + rng.uniform(0.05, 0.95)) / N_TRAIN
        frac = min(0.999999, max(0.000001, frac))
        Re = math.exp(log_lo + frac * (log_hi - log_lo))
        clean = Cd_true(Re, p, A, q, B)
        noisy = clean * math.exp(rng.gauss(0.0, NOISE_SIGMA))
        rows.append((Re, noisy))
    rows.sort(key=lambda r: r[0])
    return rows


# held-out grid: log-spaced across several decades beyond the training range,
# plus a few extreme "adversarial corner" points appended at the far end.
_HELDOUT_LOG_RE = [math.log(v) for v in
                    (60, 90, 140, 220, 340, 520, 800, 1200, 1800, 2700,
                     4000, 6000, 9000, 14000, 21000)]
_HELDOUT_CORNERS = [math.log(v) for v in (35000, 55000, 85000, 130000, 200000)]


def heldout(t):
    p, A, q, B = hidden_law(t)
    rng = random.Random(555 + t * 29)
    logs = _HELDOUT_LOG_RE + _HELDOUT_CORNERS
    clean = []
    noisy = []
    for lg in logs:
        Re = math.exp(lg)
        c = Cd_true(Re, p, A, q, B)
        n = c * math.exp(rng.gauss(0.0, HELDOUT_SIGMA))
        clean.append((Re, c))
        noisy.append((Re, n))
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
            if nm != "Re":
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
    for pre in ("cd", "cd(re)", "out", "y", "f(re)"):
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


def eval_law(code, Re):
    env = dict(_FUNCS)
    env["Re"] = float(Re)
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
    cds = [c for (_, c) in rows]
    geo_log_mean = sum(math.log(c) for c in cds) / len(cds)

    clean, noisy = heldout(t)

    se = []
    for (Re, _c), (_, ny) in zip(clean, noisy):
        pred = eval_law(code, Re)
        if pred is None or pred <= 0.0:
            fail("non-finite/non-positive prediction at Re=%.3g" % Re)
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
