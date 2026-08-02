#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the thermal-runaway-onset threshold predictor.  The
solver submits a 3-line THRESH/BELOW/ABOVE program.  This grader:

- Reads the test id from <in>, then regenerates the hidden law (A, Ta_crit)
  and the HELD-OUT grid (interpolation + near-critical + super-critical
  ambients, none shown in training) entirely from that id -- mirrors gen.py
  byte-for-byte on the shared physics.
- Parses the submission with a strict AST whitelist (THRESH: constants only;
  BELOW/ABOVE: constants + Ta).
- Rolls the piecewise predictor over the held-out grid, scores a bounded
  relative error with a small node-count parsimony penalty (maximise):
      d_i    = min(1, |p_i-T_i| / (|p_i|+|T_i|+eps))
      metric = mean_i d_i
      O      = metric * (1 + LAMBDA*nodes)
      B      = baseline_metric * (1 + LAMBDA*1)     # baseline = constant median(train T)
      Ratio  = min(1000, 100*B/O) / 1000
  A constant reproduces the baseline (~0.1).  A smooth curve fit through only
  sub-critical training rows keeps extrapolating the same shape and misses
  both the near-critical curvature and (badly) the super-critical jump to
  Tfail.  Recovering A row-wise from the given b,h (A = h*(T-Ta)*exp(-b*T))
  and comparing it against the given cooling ceiling Hmax locates the true
  threshold and lets ABOVE correctly report Tfail; held-out noise keeps even
  that below the ceiling, leaving headroom.
"""
import sys, math, ast, random

# ---- fixed design constants (mirrored byte-for-byte in gen.py) ----
B_LO, B_HI = 0.04, 0.14
H_LO, H_HI = 2.5, 7.0
HMAX_CAP_FRAC = 0.62
EW_LO, EW_HI = 3.0, 8.0
TACRIT_LO, TACRIT_HI = 28.0, 55.0
TRAIN_SPAN = 20.0
GAP_LO, GAP_HI = 1.5, 4.0
TFAIL_MARGIN_LO, TFAIL_MARGIN_HI = 20.0, 45.0
NOISE_TRAIN_FRAC = 0.10
N_TRAIN = 46

# ---- held-out / scoring constants (grader only) ----
NOISE_HELD = 0.16             # multiplicative lognormal sigma (held-out only)
N_INTERP = 10
N_NEAR = 10
N_SUPER = 10
NEAR_EPS = 0.10
SUPER_SPAN = 45.0
LAMBDA = 0.01
CAP = 1.0
MAX_NODES = 40
MAX_OUT_BYTES = 20000


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden thermal-runaway law (identical to gen.py) ----------
def params(t):
    rng = random.Random(4058000 + t * 7919003)
    b = rng.uniform(B_LO, B_HI)
    h = rng.uniform(H_LO, H_HI)
    ew_cap = HMAX_CAP_FRAC / b
    ew_hi = min(EW_HI, ew_cap)
    ew_lo = min(EW_LO, 0.8 * ew_hi)
    elbow_width = rng.uniform(ew_lo, ew_hi)
    Hmax = elbow_width * h
    Ta_crit = rng.uniform(TACRIT_LO, TACRIT_HI)
    elbow = Ta_crit + elbow_width
    A = Hmax * math.exp(-b * elbow)
    margin = rng.uniform(TFAIL_MARGIN_LO, TFAIL_MARGIN_HI)
    Tfail = elbow + margin
    gap = rng.uniform(GAP_LO, GAP_HI)
    noise_train = NOISE_TRAIN_FRAC * elbow_width
    return b, h, Hmax, A, Ta_crit, Tfail, gap, noise_train


def true_T(Ta, b, h, Hmax, A, Tfail):
    elbow = Ta + Hmax / h

    def phi(T):
        return h * (T - Ta) - A * math.exp(min(700.0, b * T))

    if phi(elbow) < 0.0:
        return Tfail
    lo, hi = Ta, elbow
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if phi(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def gen_train(t):
    b, h, Hmax, A, Ta_crit, Tfail, gap, noise_train = params(t)
    rng = random.Random(2231000 + t * 101)
    Ta_hi = Ta_crit - gap
    Ta_lo = Ta_hi - TRAIN_SPAN
    rows = []
    for _ in range(N_TRAIN):
        u = rng.random() ** 0.6
        Ta = Ta_lo + (Ta_hi - Ta_lo) * u
        Ttrue = true_T(Ta, b, h, Hmax, A, Tfail)
        Tobs = Ttrue + rng.gauss(0.0, noise_train)
        rows.append((Ta, Tobs))
    rows.sort()
    return rows, (b, h, Hmax, A, Ta_crit, Tfail, Ta_lo, Ta_hi)


def gen_held(t):
    """Held-out grid: interpolation + near-critical + super-critical ambients."""
    _, hidden = gen_train(t)
    b, h, Hmax, A, Ta_crit, Tfail, Ta_lo, Ta_hi = hidden
    rng = random.Random(9130000 + t * 31)
    pts = []
    for _ in range(N_INTERP):
        Ta = Ta_lo + (Ta_hi - Ta_lo) * rng.random()
        Ttrue = true_T(Ta, b, h, Hmax, A, Tfail)
        pts.append((Ta, Ttrue * math.exp(rng.gauss(0.0, NOISE_HELD))))
    near_lo = Ta_hi
    near_hi = Ta_crit - NEAR_EPS
    if near_hi <= near_lo:
        near_hi = near_lo + 0.05
    for _ in range(N_NEAR):
        Ta = near_lo + (near_hi - near_lo) * rng.random()
        Ttrue = true_T(Ta, b, h, Hmax, A, Tfail)
        pts.append((Ta, Ttrue * math.exp(rng.gauss(0.0, NOISE_HELD))))
    super_lo = Ta_crit + NEAR_EPS
    super_hi = Ta_crit + SUPER_SPAN
    for _ in range(N_SUPER):
        Ta = super_lo + (super_hi - super_lo) * rng.random()
        Ttrue = true_T(Ta, b, h, Hmax, A, Tfail)  # == Tfail here
        pts.append((Ta, Ttrue * math.exp(rng.gauss(0.0, NOISE_HELD))))
    return pts


# ---------- expression parsing / validation ----------
ALLOWED_FUNCS = {
    "sqrt": lambda x: math.sqrt(x),
    "log": lambda x: math.log(x),
    "exp": lambda x: math.exp(max(-700.0, min(700.0, x))),
    "sig": lambda x: 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x)))),
    "tanh": math.tanh,
    "absv": abs,
}

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
    ast.USub, ast.UAdd,
)


def _validate(tree, allowed_names):
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return "disallowed syntax %s" % type(node).__name__
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS):
                return "disallowed call"
            if node.keywords or len(node.args) != 1:
                return "bad function arity"
        if isinstance(node, ast.Name):
            if node.id in ALLOWED_FUNCS:
                continue
            if node.id in allowed_names:
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


def parse_one(text, allowed_names, tag):
    text = text.strip()
    if not text:
        fail("empty %s expression" % tag)
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("%s parse error" % tag)
    err = _validate(tree, allowed_names)
    if err:
        fail("%s: %s" % (tag, err))
    nodes = _count_nodes(tree)
    if nodes > MAX_NODES:
        fail("%s expression too large (%d nodes)" % (tag, nodes))
    try:
        code = compile(tree, "<expr>", "eval")
    except Exception:
        fail("%s compile error" % tag)
    return code, nodes


def eval_code(code, Ta=None):
    env = dict(ALLOWED_FUNCS)
    if Ta is not None:
        env["Ta"] = Ta
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


def parse_submission(raw):
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    tags = {}
    order = ["THRESH", "BELOW", "ABOVE"]
    if len(lines) < 3:
        fail("expected 3 lines THRESH/BELOW/ABOVE")
    # take the first occurrence of each required tag, in any of the first
    # few non-empty lines, but require the canonical 3-line shape.
    found = {}
    for ln in lines:
        parts = ln.split(None, 1)
        if len(parts) != 2:
            continue
        tag = parts[0].upper()
        if tag in order and tag not in found:
            found[tag] = parts[1]
    for tag in order:
        if tag not in found:
            fail("missing %s line" % tag)
    thr_code, thr_nodes = parse_one(found["THRESH"], set(), "THRESH")
    below_code, below_nodes = parse_one(found["BELOW"], {"Ta"}, "BELOW")
    above_code, above_nodes = parse_one(found["ABOVE"], {"Ta"}, "ABOVE")
    thr_val = eval_code(thr_code, Ta=None)
    if thr_val is None:
        fail("non-finite THRESH")
    total_nodes = thr_nodes + below_nodes + above_nodes
    return thr_val, below_code, above_code, total_nodes


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

    thr_val, below_code, above_code, nodes = parse_submission(text)

    held = gen_held(t)
    ds = []
    for Ta, T in held:
        code = below_code if Ta < thr_val else above_code
        p = eval_code(code, Ta=Ta)
        if p is None:
            fail("non-finite / invalid prediction")
        d = abs(p - T) / (abs(p) + abs(T) + 1e-6)
        ds.append(min(CAP, d))
    metric = sum(ds) / len(ds)

    # baseline: constant predictor = median of TRAIN T
    train, _ = gen_train(t)
    Ts = sorted(r[1] for r in train)
    n = len(Ts)
    med = Ts[n // 2] if n % 2 else 0.5 * (Ts[n // 2 - 1] + Ts[n // 2])
    bd = [min(CAP, abs(med - T) / (abs(med) + abs(T) + 1e-6)) for _, T in held]
    Bmetric = sum(bd) / len(bd)

    B = Bmetric * (1.0 + LAMBDA * 1)
    O = metric * (1.0 + LAMBDA * nodes)
    sc = min(1000.0, 100.0 * B / max(1e-12, O))
    print("metric=%.6f baseline=%.6f nodes=%d thresh=%.6f  Ratio: %.6f"
          % (metric, Bmetric, nodes, thr_val, sc / 1000.0))


if __name__ == "__main__":
    main()
