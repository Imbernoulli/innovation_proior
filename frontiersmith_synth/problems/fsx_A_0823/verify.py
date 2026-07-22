#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the "two hidden tuning forks in one box" coupled-mode
resonance task.  The solver submits ONE closed-form expression for the
response AMPLITUDE |X1| as a function of the drive frequency w.

- Reads the test id from <in>'s header, then regenerates the hidden physical
  constants (Om1, Om2, g1, g2, kc) and the HELD-OUT EXTRAPOLATION grid
  entirely from that id (identical formulas to gen.py, duplicated here).  The
  law and its coefficients live ONLY inside this checker and gen.py -- never
  in a shared importable module, never in the training data.
- The held-out grid has four bands: a fresh same-band interpolation check (a
  fair test on the ONE regime the solver actually saw), a modest tail
  extension, the full resonance region (including the avoided-crossing
  double peak between Om1 and Om2), and the high-frequency asymptotic roll
  off (|X1| ~ 1/w^4 for the true two-pole-pair system) -- three genuinely
  novel regimes plus one honest one.
- Parses the submitted expression with a strict AST whitelist:
      name      w
      operators + - * / **  and unary +/-
      functions sqrt absv
      numeric constants
- Evaluates it on the held-out grid, computes a bounded symmetric relative
  error per point, averages, and adds a small node-count parsimony penalty
  (minimise):
      metric = mean_i min(1, |p_i - t_i| / (|p_i| + |t_i|))
      O = metric * (1 + LAMBDA * nodes)
      B = baseline_metric * (1 + LAMBDA * 1)   # baseline = constant predictor
                                                #   = mean(train amplitude)
      Ratio = min(1000, 100 * B / O) / 1000
  A constant predictor reproduces the baseline (~0.1).  A single-resonance
  ("one tuning fork") fit matches the smooth training tail just as well --
  the tail is featureless -- but it has only ONE pole pair, so it predicts
  ONE peak where the true two-mode system has TWO split peaks (an avoided
  crossing) and the wrong high-frequency power law (1/w^2, not 1/w^4).  Only
  a predictor that COMMITS to the physical two-mode rational form (quartic
  denominator, quadratic numerator) and solves for its five constants
  survives extrapolation into the held-out regions.  Held-out noise keeps
  even a correct discovery below the ceiling, leaving headroom.
"""
import sys, math, ast, random

# ---- fixed design constants (mirrored byte-for-byte in gen.py) ----
OM1_LO, OM1_HI = 4.0, 5.0
RATIO_LO, RATIO_HI = 1.25, 1.70
DAMP_FRAC_LO, DAMP_FRAC_HI = 0.06, 0.16
COUPL_FRAC_LO, COUPL_FRAC_HI = 0.04, 0.15
TAIL_FMIN_FRAC = 0.05
TAIL_FMAX_FRAC = 0.45
N_TRAIN = 44
NOISE_TRAIN = 0.05

# ---- held-out / scoring constants (grader only) ----
# Four regimes, none of them the exact training draw: a fresh same-band
# interpolation check (M_INTERP), a short tail extension (M_TAIL), the full
# resonance / avoided-crossing region (M_RES), and the high-frequency
# asymptote (M_ASY). Held-out lognormal noise (NOISE_HELD) leaves irreducible
# error even for a perfect model.
NOISE_HELD = 0.10
M_INTERP, M_TAIL, M_RES, M_ASY = 50, 15, 15, 20
LAMBDA = 0.0025
CAP = 1.0
MAX_NODES = 140
MAX_OUT_BYTES = 100000

ALLOWED_FUNCS = {
    "sqrt": lambda x: math.sqrt(x),
    "absv": abs,
}
ALLOWED_NAMES = {"w"}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden physical law (identical to gen.py) ----------
def hidden_params(t):
    rng = random.Random(700000 + t * 913171)
    Om1 = rng.uniform(OM1_LO, OM1_HI)
    ratio = rng.uniform(RATIO_LO, RATIO_HI)
    Om2 = Om1 * ratio
    g1 = rng.uniform(DAMP_FRAC_LO, DAMP_FRAC_HI) * Om1
    g2 = rng.uniform(DAMP_FRAC_LO, DAMP_FRAC_HI) * Om2
    cfrac = rng.uniform(COUPL_FRAC_LO, COUPL_FRAC_HI)
    kc = cfrac * (Om1 * Om1)
    return Om1, Om2, g1, g2, kc


def response(w, Om1, Om2, g1, g2, kc):
    a1 = Om1 * Om1 - w * w
    a2 = Om2 * Om2 - w * w
    b1 = g1 * w
    b2 = g2 * w
    Dre = a1 * a2 - b1 * b2 - kc * kc
    Dim = a1 * b2 + a2 * b1
    Dmag2 = Dre * Dre + Dim * Dim
    if Dmag2 < 1e-300:
        Dmag2 = 1e-300
    Xre = (a2 * Dre + b2 * Dim) / Dmag2
    Xim = (b2 * Dre - a2 * Dim) / Dmag2
    return Xre, Xim


def amplitude(Xre, Xim):
    return math.sqrt(Xre * Xre + Xim * Xim)


def gen_train(t):
    Om1, Om2, g1, g2, kc = hidden_params(t)
    rng = random.Random(271828 + t * 97)
    fmin = TAIL_FMIN_FRAC * Om1
    fmax = TAIL_FMAX_FRAC * Om1
    Xre0, Xim0 = response(fmax, Om1, Om2, g1, g2, kc)
    scale0 = amplitude(Xre0, Xim0)
    rows = []
    for _ in range(N_TRAIN):
        w = rng.uniform(fmin, fmax)
        Xre, Xim = response(w, Om1, Om2, g1, g2, kc)
        Xre_n = Xre + NOISE_TRAIN * scale0 * rng.gauss(0.0, 1.0)
        Xim_n = Xim + NOISE_TRAIN * scale0 * rng.gauss(0.0, 1.0)
        rows.append((w, Xre_n, Xim_n))
    rows.sort()
    return rows


def gen_held(t):
    """Held-out grid: a fresh same-band interpolation check (rewards a fit
    that is honestly good on the ONE regime it was trained on), a modest
    tail extension, the FULL resonance region (both peaks + the
    avoided-crossing dip between them, never shown in training), and the
    high-frequency asymptotic roll-off -- three genuinely novel regimes a
    tail-only fit never sees, plus one fair one."""
    Om1, Om2, g1, g2, kc = hidden_params(t)
    rng = random.Random(555001 + t * 777001)
    fmin = TAIL_FMIN_FRAC * Om1
    fmax = TAIL_FMAX_FRAC * Om1
    segments = [
        (fmin, fmax, M_INTERP),
        (fmax, 0.80 * Om1, M_TAIL),
        (0.80 * Om1, 1.30 * Om2, M_RES),
        (1.70 * Om2, 3.20 * Om2, M_ASY),
    ]
    pts = []
    for lo, hi, m in segments:
        for _ in range(m):
            w = rng.uniform(lo, hi)
            Xre, Xim = response(w, Om1, Om2, g1, g2, kc)
            a_true = amplitude(Xre, Xim)
            a_meas = a_true * math.exp(rng.gauss(0.0, NOISE_HELD))
            pts.append((w, a_meas))
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


def eval_at(code, w):
    env = dict(ALLOWED_FUNCS)
    env["w"] = w
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
    for w, a_true in held:
        p = eval_at(code, w)
        if p is None:
            fail("non-finite / invalid prediction")
        d = abs(p - a_true) / (abs(p) + abs(a_true) + 1e-30)
        ds.append(min(CAP, d))
    metric = sum(ds) / len(ds)

    # baseline: constant predictor = mean training amplitude
    train = gen_train(t)
    amps = [amplitude(Xre, Xim) for _, Xre, Xim in train]
    const_pred = sum(amps) / len(amps)
    bd = [min(CAP, abs(const_pred - a_true) / (abs(const_pred) + abs(a_true) + 1e-30))
          for _, a_true in held]
    Bmetric = sum(bd) / len(bd)

    B = Bmetric * (1.0 + LAMBDA * 1)
    O = metric * (1.0 + LAMBDA * nodes)
    sc = min(1000.0, 100.0 * B / max(1e-12, O))
    print("metric=%.6f baseline=%.6f nodes=%d  Ratio: %.6f"
          % (metric, Bmetric, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
