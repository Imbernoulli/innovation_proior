#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the "demand cannibalize forecast" task.

- Reads the test id from <in>, then regenerates the hidden scenario
  (A1, k, M0, t_B, s_true) entirely from that id -- IDENTICAL logic to
  gen.py's hidden_law()/simulate(), duplicated here on purpose (no shared
  importable module) so nothing about the law is reachable from the solver's
  sandbox.
- Regenerates the HELD-OUT window t = N_TRAIN+1 .. N_TRAIN+H (strictly past
  the training window, straddling the competitor's launch step t_B) with a
  fresh noise draw. This window is never shown to the solver.
- Parses the participant's closed-form adoption law: a single Python
  expression in the variable `t`, optionally ONE top-level conditional
  ("EXPR1 if t < C else EXPR2" / with <=, >, >=), constants, + - * /, unary
  +/-, and the functions expv(a), logv(a), sqrtv(a), absv(a), minv(a,b),
  maxv(a,b), powv(a,b) [a must evaluate > 0].
- Scores by mean SQUARED LOG ERROR between the law's prediction and the
  (noisy) held-out truth, with a small node-count parsimony penalty:
      F = mean_k (log(pred_k) - log(true_noisy_k))^2 * (1 + LAMBDA*nodes)
      B = mean_k (log(A_last)  - log(true_noisy_k))^2 * (1 + LAMBDA*1)
      Ratio = min(CAP, 0.1 * (B/F) ** GAMMA)
  The baseline B freezes adoption at its last TRAINING value forever (the
  "growth has already happened" naive forecast). Squared LOG error rewards
  matching the trajectory's shape/rate across a window that can span a
  couple of x-fold changes in level, not just the level at one point. GAMMA
  compresses B/F sub-linearly so a merely-right-shaped law doesn't saturate;
  CAP < 1 keeps the ceiling open even for a law that nails the shape. Held-out
  observation noise plus the discrete-vs-analytic mismatch between the true
  recursion and any smooth closed form keep even a strong law below CAP.
"""
import sys, math, ast, random

N_TRAIN = 13
H = 11
NOISE_SIGMA = 0.015
HELDOUT_SIGMA = 0.095

LAMBDA = 0.010
GAMMA = 0.40
CAP = 0.78
MAX_NODES = 200
MAX_OUT_BYTES = 200000
CLAMP_LOG = 60.0

ALLOWED_FUNCS_ARITY = {"expv": 1, "logv": 1, "sqrtv": 1, "absv": 1,
                        "minv": 2, "maxv": 2, "powv": 2}

PLAN = {
    1: dict(k=0.22, M0=7200.0, tB_off=9, s=0.40),
    2: dict(k=0.24, M0=6600.0, tB_off=8, s=0.44),
    3: dict(k=0.40, M0=5200.0, tB_off=3, s=0.72),
    4: dict(k=0.26, M0=7800.0, tB_off=9, s=0.36),
    5: dict(k=0.42, M0=4800.0, tB_off=3, s=0.75),
    6: dict(k=0.24, M0=8200.0, tB_off=8, s=0.48),
    7: dict(k=0.25, M0=6000.0, tB_off=7, s=0.40),
    8: dict(k=0.38, M0=5500.0, tB_off=4, s=0.67),
    9: dict(k=0.20, M0=7000.0, tB_off=9, s=0.40),
    10: dict(k=0.26, M0=6800.0, tB_off=7, s=0.49),
}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden law (IDENTICAL to gen.py) ----------
def hidden_law(t):
    rng = random.Random(119600 + t * 7919)
    plan = PLAN.get(t)
    if plan is None:
        fail("bad test id")
    A1 = rng.uniform(90.0, 140.0)
    k = plan['k'] + rng.uniform(-0.01, 0.01)
    M0 = plan['M0'] + rng.uniform(-150.0, 150.0)
    tB = N_TRAIN + plan['tB_off']
    s_true = plan['s'] + rng.uniform(-0.02, 0.02)
    s_hint = s_true * (1.0 + rng.uniform(-0.15, 0.15)) + rng.uniform(-0.03, 0.03)
    s_hint = max(0.05, min(0.95, s_hint))
    M_hint = M0 * (1.0 + rng.uniform(-0.12, 0.12))
    return A1, k, M0, tB, s_true, s_hint, M_hint


def simulate(A1, k, M0, tB, s_true, T_end):
    A = {1: A1}
    M1 = None
    for t in range(1, T_end):
        cur_cap = M0 if (t < tB) else M1
        nxt = A[t] + k * A[t] * (1.0 - A[t] / cur_cap)
        A[t + 1] = nxt
        if (t + 1) == tB:
            M1 = A[tB] + (1.0 - s_true) * (M0 - A[tB])
    return A


def train_rows(t):
    A1, k, M0, tB, s_true, s_hint, M_hint = hidden_law(t)
    A = simulate(A1, k, M0, tB, s_true, N_TRAIN)
    rng = random.Random(220300 + t * 13)
    rows = []
    for ti in range(1, N_TRAIN + 1):
        noisy = A[ti] * math.exp(rng.gauss(0.0, NOISE_SIGMA))
        rows.append((ti, noisy))
    return rows


def heldout(t):
    A1, k, M0, tB, s_true, s_hint, M_hint = hidden_law(t)
    A = simulate(A1, k, M0, tB, s_true, N_TRAIN + H)
    rng = random.Random(778800 + t * 29)
    clean, noisy = [], []
    for ti in range(N_TRAIN + 1, N_TRAIN + H + 1):
        c = A[ti]
        n = c * math.exp(rng.gauss(0.0, HELDOUT_SIGMA))
        clean.append((ti, c))
        noisy.append((ti, n))
    return clean, noisy


# ---------- expression parsing / validation ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
    ast.IfExp, ast.Compare, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)


def _validate_ast(tree):
    n_compare = 0
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
            if nm != "t":
                return "unknown name %s" % nm
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                return "non-numeric constant"
            v = float(node.value)
            if v != v or v in (float("inf"), float("-inf")):
                return "non-finite constant"
        if isinstance(node, ast.Compare):
            n_compare += 1
            if len(node.ops) != 1 or len(node.comparators) != 1:
                return "compound comparison not allowed"
            if not isinstance(node.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                return "disallowed comparison operator"
    if n_compare > 1:
        return "at most one conditional allowed"
    return None


def _count_nodes(tree):
    return sum(1 for nd in ast.walk(tree)
               if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name,
                                   ast.Constant, ast.Compare, ast.IfExp)))


def parse_law(raw):
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        fail("empty output")
    text = lines[-1]
    low = text.lower()
    for pre in ("a(t)", "a", "y", "f(t)"):
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
        fail("law too large (%d nodes)" % nodes)
    try:
        code = compile(tree, "<law>", "eval")
    except Exception:
        fail("compile error")
    return code, nodes


def _guard1(fn):
    def g(a):
        if not isinstance(a, (int, float)) or isinstance(a, bool):
            return float("nan")
        try:
            return fn(a)
        except Exception:
            return float("nan")
    return g


def _logv(a):
    if a <= 0.0:
        return float("nan")
    return math.log(a)


def _sqrtv(a):
    if a < 0.0:
        return float("nan")
    return math.sqrt(a)


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


_FUNCS = {
    "expv": _guard1(math.exp),
    "logv": _guard1(_logv),
    "sqrtv": _guard1(_sqrtv),
    "absv": _guard1(abs),
    "minv": _minv,
    "maxv": _maxv,
    "powv": _powv,
}


def eval_law(code, t):
    env = dict(_FUNCS)
    env["t"] = float(t)
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
        tid = int(header[0])
    except Exception:
        fail("bad instance header")
    if tid not in PLAN:
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

    rows = train_rows(tid)
    A_last = rows[-1][1]

    clean, noisy = heldout(tid)

    se = []
    for (ti, _c), (_, ny) in zip(clean, noisy):
        pred = eval_law(code, ti)
        if pred is None or pred <= 0.0:
            fail("non-finite/non-positive prediction at t=%d" % ti)
        lp = max(-CLAMP_LOG, min(CLAMP_LOG, math.log(pred)))
        lt = math.log(ny)
        se.append((lp - lt) ** 2)
    F_mse = sum(se) / len(se)

    se_b = [(math.log(A_last) - math.log(ny)) ** 2 for (_, ny) in noisy]
    B_mse = sum(se_b) / len(se_b)

    F = F_mse * (1.0 + LAMBDA * nodes)
    B = B_mse * (1.0 + LAMBDA * 1)
    ratio_raw = B / max(1e-9, F)
    sc = min(CAP, 0.1 * (ratio_raw ** GAMMA))
    print("heldout_MSLE=%.6f baseline_MSLE=%.6f nodes=%d B/F=%.4f  Ratio: %.6f"
          % (F_mse, B_mse, nodes, ratio_raw, sc))


if __name__ == "__main__":
    main()
