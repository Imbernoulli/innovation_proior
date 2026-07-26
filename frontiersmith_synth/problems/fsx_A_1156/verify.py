#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the harbor tide-clock commensurability task.  The
solver submits ONE closed-form expression for the tide height y as a function
of time t.

- Reads the case id from <in> (header), then regenerates the hidden tide law
  and the HELD-OUT EXTRAPOLATION window (several locked super-periods after
  the training window) entirely from that id. The law and its coefficients
  live ONLY here.
- Parses the submitted expression with a strict AST whitelist:
      name       t
      operators  + - * / **  and unary +/-
      functions  sin cos
      numeric constants
- Evaluates it on the held-out window, computes a bounded symmetric relative
  error per point, averages, and adds a small node-count parsimony penalty
  (minimise):
      metric = mean_i min(1, |p_i - t_i| / (|p_i| + |t_i| + eps))
      O = metric * (1 + LAMBDA * nodes)
      B = baseline_metric * (1 + LAMBDA * 1)     # baseline = constant train mean
      Ratio = min(1000, 100 * B / O) / 1000
  A constant predictor reproduces the baseline (~0.1). Treating the four
  gears as independent frequencies fits the training window (which only
  spans PART of one locked super-period, so the three locked gears sit
  closer together than the window's own frequency resolution and are not
  independently identifiable) but the fitted frequencies drift out of phase
  by the held-out window, several super-periods later -- only recovering the
  EXACT shared integer relationship among three of the four gears keeps
  phase coherent that far out. Held-out sensor noise keeps even the correct
  law below the ceiling, leaving headroom.
"""
import sys, math, ast, random

# ---- fixed design constants (mirrored byte-for-byte in gen.py) ----
CANDIDATE_TRIPLES = [(2, 3, 7), (3, 4, 5), (2, 5, 7), (3, 5, 8),
                      (2, 3, 11), (4, 5, 7), (2, 7, 9), (3, 7, 8)]
F0_LO, F0_HI = 0.0045, 0.0065
A_LO, A_HI = 0.6, 1.4
F4_LO, F4_HI = 0.16, 0.30
NOISE_TRAIN = 0.08
DT_TRAIN = 0.5
T_TRAIN_LO, T_TRAIN_HI = 120.0, 190.0

# ---- held-out / scoring constants (grader only) ----
NOISE_HELD = 0.28
DT_HELD = 0.5
T_HELD = 60.0
HELD_OFFSET = 450.0            # several locked super-periods after training
LAMBDA = 0.002
CAP = 1.0
MAX_NODES = 120
MAX_OUT_BYTES = 100000
EPS = 1e-6

ALLOWED_FUNCS = {"sin": math.sin, "cos": math.cos}
ALLOWED_NAMES = {"t"}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden tide law (identical to gen.py) ----------
def n_train_for(t):
    frac = max(0.0, min(1.0, (t - 1) / 9.0))
    span = T_TRAIN_HI - frac * (T_TRAIN_HI - T_TRAIN_LO)
    return int(round(span / DT_TRAIN))


def params(t):
    rng = random.Random(424242 + t * 97531)
    n1, n2, n3 = rng.choice(CANDIDATE_TRIPLES)
    f0 = rng.uniform(F0_LO, F0_HI)
    A = [rng.uniform(A_LO, A_HI) for _ in range(4)]
    phi = [rng.uniform(0.0, 2 * math.pi) for _ in range(4)]
    while True:
        f4 = rng.uniform(F4_LO, F4_HI)
        ok = True
        for m in range(1, 26):
            if abs(f4 - m * f0) < 0.02:
                ok = False
                break
        if ok:
            break
    return n1, n2, n3, f0, f4, A, phi


def true_y(tt, n1, n2, n3, f0, f4, A, phi):
    w1, w2, w3, w4 = 2 * math.pi * n1 * f0, 2 * math.pi * n2 * f0, 2 * math.pi * n3 * f0, 2 * math.pi * f4
    return (A[0] * math.sin(w1 * tt + phi[0]) + A[1] * math.sin(w2 * tt + phi[1]) +
            A[2] * math.sin(w3 * tt + phi[2]) + A[3] * math.sin(w4 * tt + phi[3]))


def gen_train(t):
    n1, n2, n3, f0, f4, A, phi = params(t)
    n_train = n_train_for(t)
    rng = random.Random(1111 + t * 13)
    rows = []
    for i in range(n_train):
        tt = i * DT_TRAIN
        y = true_y(tt, n1, n2, n3, f0, f4, A, phi) + rng.gauss(0.0, NOISE_TRAIN)
        rows.append((tt, y))
    return rows


def gen_held(t):
    n1, n2, n3, f0, f4, A, phi = params(t)
    n_held = int(round(T_HELD / DT_HELD))
    rng = random.Random(9999 + t * 7)
    rows = []
    for i in range(n_held):
        tt = HELD_OFFSET + i * DT_HELD
        y = true_y(tt, n1, n2, n3, f0, f4, A, phi) + rng.gauss(0.0, NOISE_HELD)
        rows.append((tt, y))
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


def eval_at(code, tt):
    env = dict(ALLOWED_FUNCS)
    env["t"] = tt
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
    for tt, y in held:
        p = eval_at(code, tt)
        if p is None:
            fail("non-finite / invalid prediction")
        d = abs(p - y) / (abs(p) + abs(y) + EPS)
        ds.append(min(CAP, d))
    metric = sum(ds) / len(ds)

    # baseline: constant predictor = arithmetic mean of TRAIN y
    train = gen_train(t)
    ybar = sum(r[1] for r in train) / len(train)
    bd = [min(CAP, abs(ybar - y) / (abs(ybar) + abs(y) + EPS)) for _, y in held]
    Bmetric = sum(bd) / len(bd)

    B = Bmetric * (1.0 + LAMBDA * 1)
    O = metric * (1.0 + LAMBDA * nodes)
    sc = min(1000.0, 100.0 * B / max(1e-12, O))
    print("metric=%.6f baseline=%.6f nodes=%d  Ratio: %.6f"
          % (metric, Bmetric, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
