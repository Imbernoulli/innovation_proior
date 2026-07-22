#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the dimensionless-grouping calibration task.
The solver submits ONE closed-form Python expression for the response
y as a function of the four knobs x1..x4.

- Reads the case id from <in> (header), then regenerates the hidden
  grading matrix U, the unique dimensionless-group exponent vector b,
  the scaling exponent p, amplitude C, and the HELD-OUT EXTRAPOLATION
  grid entirely from that id (identical code path to gen.py for the
  training side). The law and its coefficients live ONLY here.
- Parses the submitted expression with a strict AST whitelist:
      names     x1 x2 x3 x4
      operators + - * / **  and unary +/-
      functions sqrt log exp absv
      numeric constants (finite only)
- Evaluates it on the held-out grid, computes a bounded symmetric
  relative error per point, averages, and adds a small node-count
  parsimony penalty (minimise):
      metric = mean_i min(1, |p_i - t_i| / (|p_i| + |t_i|))
      O = metric * (1 + LAMBDA * nodes)
      B = baseline_metric * (1 + LAMBDA * 1)   # baseline = constant geomean(train y)
      Ratio = min(1000, 100 * B / O) / 1000
  A constant predictor reproduces the baseline (~0.1). A generic
  per-knob power-law regression (fit log y ~ c0 + sum a_i log x_i
  freely, ignoring the grading matrix) matches the training band
  (narrow, near-orthogonal design) reasonably well IN-SAMPLE, but its
  four fitted exponents pick up noise in ALL FOUR raw directions, not
  just the one direction that the grading matrix says actually matters.
  Most held-out points are pushed along a mix of the true dimensionless
  direction and a large "wasted" motion orthogonal to it (zero true
  effect on y, but large motion in raw-knob space) -- the raw fit's
  noise in the orthogonal directions gets amplified by that large
  motion and the prediction drifts far from the truth. Only a predictor
  that first computes the exact null-space direction b from U (linear
  algebra, no fitting needed for the direction) and fits ONLY the
  single scalar exponent p along it is immune to that contamination.
"""
import sys, math, ast, random

# ---- fixed design constants (mirrored byte-for-byte in gen.py) ----
BASE_SEED_PARAMS = 730100
MULT_PARAMS = 91121
BASE_SEED_TRAIN = 550301
MULT_TRAIN = 4021

W_TRAIN = math.log(1.30)
NOISE_TRAIN = 0.04
N_TRAIN = 20

M_LO, M_HI = 2.0, 9.0
P_ABS_LO, P_ABS_HI = 0.5, 2.0
C_LO, C_HI = 0.8, 3.5

# ---- held-out / scoring constants (grader only) ----
BASE_SEED_HELD = 881003
MULT_HELD = 6173
NOISE_HELD = 0.35
N_HELD = 60
ORTHO_PROB = 0.75          # fraction of held-out points using the "wasted motion" trap
DELTA_LO, DELTA_HI = 1.0, 2.2      # along-b log-shift magnitude (both branches)
PERP_LO, PERP_HI = 3.5, 7.0        # orthogonal "wasted" log-shift magnitude (trap branch only)
LAMBDA = 0.004
CAP = 1.0
MAX_NODES = 60
MAX_OUT_BYTES = 100000

ALLOWED_FUNCS = {
    "sqrt": lambda x: math.sqrt(x),
    "log":  lambda x: math.log(x),
    "exp":  lambda x: math.exp(max(-700.0, min(700.0, x))),
    "absv": abs,
}
ALLOWED_NAMES = {"x1", "x2", "x3", "x4"}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden grading matrix + scaling law (identical to gen.py) ----------
def _det3(R):
    return (R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1])
            - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
            + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]))


def params(t):
    rng = random.Random(BASE_SEED_PARAMS + t * MULT_PARAMS)

    while True:
        b = [rng.randint(-2, 2) for _ in range(4)]
        if all(v != 0 for v in b):
            break
    b4 = b[3]

    while True:
        R = [[rng.randint(-3, 3) for _ in range(3)] for _ in range(3)]
        if _det3(R) != 0:
            break

    U = []
    for j in range(3):
        r = R[j]
        S = r[0] * b[0] + r[1] * b[1] + r[2] * b[2]
        U.append([b4 * r[0], b4 * r[1], b4 * r[2], -S])

    m = [rng.uniform(M_LO, M_HI) for _ in range(4)]

    p_mag = rng.uniform(P_ABS_LO, P_ABS_HI)
    p_mag = round(p_mag * 4) / 4.0
    if p_mag < 0.25:
        p_mag = 0.25
    p = p_mag if rng.random() < 0.5 else -p_mag

    C = rng.uniform(C_LO, C_HI)
    return U, b, m, p, C


def pi_group(x, b):
    val = 1.0
    for xi, bi in zip(x, b):
        val *= xi ** bi
    return val


def true_y(x, b, p, C):
    return C * (pi_group(x, b) ** p)


def gen_train(t):
    U, b, m, p, C = params(t)
    rng = random.Random(BASE_SEED_TRAIN + t * MULT_TRAIN)
    rows = []
    for _ in range(N_TRAIN):
        x = [m[i] * math.exp(rng.uniform(-W_TRAIN, W_TRAIN)) for i in range(4)]
        y = true_y(x, b, p, C) * math.exp(rng.gauss(0.0, NOISE_TRAIN))
        rows.append((x, y))
    return rows


def gen_held(t):
    """Held-out grid, regenerated only here. Each point moves along a
    direction d in log-knob-space built as d = (Delta/|b|^2)*b + d_perp,
    where d_perp is orthogonal to b (b . d_perp = 0 exactly). The
    dimensionless group Pi only depends on b . d = Delta, so d_perp is
    pure "wasted" motion in raw-knob space with ZERO effect on the true
    y -- but a raw per-knob fit cannot distinguish it from real signal."""
    U, b, m, p, C = params(t)
    rng = random.Random(BASE_SEED_HELD + t * MULT_HELD)
    b2 = float(sum(v * v for v in b))
    pts = []
    for _ in range(N_HELD):
        delta = rng.uniform(DELTA_LO, DELTA_HI)
        if rng.random() < 0.5:
            delta = -delta
        along = [(delta / b2) * bi for bi in b]
        if rng.random() < ORTHO_PROB:
            v = [rng.gauss(0.0, 1.0) for _ in range(4)]
            vb = sum(v[i] * b[i] for i in range(4))
            vperp = [v[i] - (vb / b2) * b[i] for i in range(4)]
            norm = math.sqrt(sum(c * c for c in vperp))
            if norm < 1e-9:
                d = along
            else:
                mag = rng.uniform(PERP_LO, PERP_HI)
                d = [along[i] + (mag / norm) * vperp[i] for i in range(4)]
        else:
            d = along
        x = [m[i] * math.exp(d[i]) for i in range(4)]
        y = true_y(x, b, p, C) * math.exp(rng.gauss(0.0, NOISE_HELD))
        pts.append((x, y))
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
            if abs(v) > 1e12:
                return "constant out of range"
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


def eval_at(code, x):
    env = dict(ALLOWED_FUNCS)
    env["x1"] = x[0]; env["x2"] = x[1]; env["x3"] = x[2]; env["x4"] = x[3]
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
    for x, y in held:
        pr = eval_at(code, x)
        if pr is None:
            fail("non-finite / invalid prediction")
        d = abs(pr - y) / (abs(pr) + abs(y) + 1e-30)
        ds.append(min(CAP, d))
    metric = sum(ds) / len(ds)

    # baseline: constant predictor = geometric mean of TRAIN y
    train = gen_train(t)
    gm = math.exp(sum(math.log(r[1]) for r in train) / len(train))
    bd = [min(CAP, abs(gm - y) / (abs(gm) + abs(y) + 1e-30)) for _, y in held]
    Bmetric = sum(bd) / len(bd)

    B = Bmetric * (1.0 + LAMBDA * 1)
    O = metric * (1.0 + LAMBDA * nodes)
    sc = min(1000.0, 100.0 * B / max(1e-12, O))
    print("metric=%.6f baseline=%.6f nodes=%d  Ratio: %.6f"
          % (metric, Bmetric, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
