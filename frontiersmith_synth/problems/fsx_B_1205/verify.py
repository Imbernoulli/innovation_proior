#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the crop-yield stress forecast.

- Reads testId from <in>'s header, then regenerates -- entirely from
  testId, identically to gen.py -- the hidden field parameters (Y0, the
  GDD sensitivity a, the flowering heat-sensitivity BETA), the SAME
  N_TRAIN historical seasons shown to the solver, and a HELD-OUT SEASON
  (never shown to the solver): an otherwise-ordinary season (climate
  anomaly w=0) with a heat wave whose timing is chosen, by a deterministic
  grid search, to land squarely on that season's OWN flowering window --
  an adversarial "heat-wave-at-the-wrong-moment" trap for anyone who only
  regresses on total seasonal heat.
- Parses the participant's output: ONE arithmetic expression string over
  the two variables G (season growing-degree-days) and H (flowering-
  window heat exceedance), using +, -, *, /, **, parentheses, numeric
  constants and a small whitelist of unary math functions. No
  conditionals, comparisons, other names, or other syntax are allowed.
- Evaluates the expression at every held-out (G, H) row, rejecting any
  non-finite value.
- Scores by held-out MSE against an internal baseline predictor: the
  simple least-squares fit of yield on G ALONE over the (regenerated)
  training seasons -- "season-total heat predicts yield; ignore exactly
  WHEN within the season any extreme heat fell" -- the naive status-quo
  forecast the theme is named for:
      F = held-out MSE of the submission
      B = held-out MSE of the G-only baseline regression
      Ratio = min(920, 100*(eps+B)/(eps+F)) / 1000
  A predictor that reproduces the baseline scores ~0.1. One that folds in
  H linearly does somewhat better. One that also applies the given
  quadratic flowering-stress penalty (using the supplied BETA) does best
  of all -- but the ratio is hard-capped at 0.92 so even a strong fit
  (bounded by season-to-season measurement noise) leaves headroom above
  the reference solution.
"""
import sys, math, ast, random

L = 100
TBASE = 10.0
TCAP = 30.0
THEAT = 32.3
FLOWER_DURATION = 12
BASE_FLOWER_START = 30.0
WARMTH_SHIFT_COEF = 14.0
N_TRAIN = 60

N_HOLDOUT = 24
WIDTH_MULT = 1.8
STORM_GRID = 800
EPS = 1e-3
CAP = 920.0
MAX_OUT_BYTES = 20000
MAX_NODES = 60
MAX_LITERAL = 1e7


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden field parameters (identical to gen.py) ----------
def instance_params(testId):
    rng = random.Random(40000 + testId * 7919)
    Y0 = 40.0 + 6.0 * rng.random()
    a_gdd = 0.018 + 0.004 * rng.random()
    BETA = 0.024 + 0.0044 * testId + 0.0007 * rng.random()
    Tmid = 21.0 + 2.0 * rng.random()
    Tamp = 8.0 + 2.0 * rng.random()
    Wscale = 3.0 + 0.5 * rng.random()
    day_noise_std = 0.8 + 0.2 * rng.random()
    calm_bump_amp = 1.0 + 0.7 * rng.random()
    calm_bump_width = 4.0 + 2.0 * rng.random()
    storm_bump_amp = 3.8 + 0.24 * testId + 0.3 * rng.random()
    storm_bump_width = 3.0 + 1.0 * rng.random()
    train_noise_std = 1.0
    holdout_noise_std = 1.4
    return dict(Y0=Y0, a_gdd=a_gdd, BETA=BETA, Tmid=Tmid, Tamp=Tamp,
                Wscale=Wscale, day_noise_std=day_noise_std,
                calm_bump_amp=calm_bump_amp, calm_bump_width=calm_bump_width,
                storm_bump_amp=storm_bump_amp, storm_bump_width=storm_bump_width,
                train_noise_std=train_noise_std, holdout_noise_std=holdout_noise_std)


def flower_window(w, p):
    fs = BASE_FLOWER_START - WARMTH_SHIFT_COEF * w
    fs = int(round(fs))
    fs = max(1, min(L - FLOWER_DURATION + 1, fs))
    fe = fs + FLOWER_DURATION - 1
    return fs, fe


def gdd_and_flower_exceed(tmax_of_day, fs, fe):
    gdd = 0.0
    flower_exc = 0.0
    for d in range(1, L + 1):
        Tm = tmax_of_day(d)
        gdd += max(0.0, min(Tm, TCAP) - TBASE)
        if fs <= d <= fe:
            flower_exc += max(0.0, Tm - THEAT)
    return gdd, flower_exc


def train_row(rng, p):
    w = rng.uniform(-1.0, 1.0)
    bump_center = rng.uniform(1.0, L)
    day_noise = [rng.gauss(0.0, p['day_noise_std']) for _ in range(L)]

    def tmax(d):
        base = (p['Tmid'] + p['Tamp'] * math.sin(math.pi * d / L)
                + w * p['Wscale'] + day_noise[d - 1])
        bump = p['calm_bump_amp'] * math.exp(-((d - bump_center) ** 2) / (2.0 * p['calm_bump_width'] ** 2))
        return base + bump

    fs, fe = flower_window(w, p)
    gdd, flower_exc = gdd_and_flower_exceed(tmax, fs, fe)
    y = p['Y0'] + p['a_gdd'] * gdd - p['BETA'] * (flower_exc ** 2) + rng.gauss(0.0, p['train_noise_std'])
    return gdd, flower_exc, y


def regenerate_training(testId, p):
    rng = random.Random(50000 + testId * 104729)
    return [train_row(rng, p) for _ in range(N_TRAIN)]


def ols_g_only(rows):
    """Fit y = a0 + a1*G on the training rows, ignoring H entirely --
    the 'season-total heat predicts yield' naive baseline the theme
    is named for."""
    n = len(rows)
    sG = sY = sGG = sGY = 0.0
    for G, H, y in rows:
        sG += G; sY += y; sGG += G * G; sGY += G * y
    denom = n * sGG - sG * sG
    if abs(denom) < 1e-9:
        return sum(y for _, _, y in rows) / n, 0.0
    a1 = (n * sGY - sG * sY) / denom
    a0 = (sY - a1 * sG) / n
    return a0, a1


def storm_center(fs, fe, p):
    """Deterministic grid search (pure arithmetic, no RNG) for the storm
    center that maximises flowering-window heat exceedance in an
    otherwise ordinary (w=0) season -- the adversarial 'wrong moment'."""
    best_c, best_exc = 1.0, -1.0
    for i in range(STORM_GRID):
        c = 1.0 + (L - 1.0) * i / (STORM_GRID - 1)

        def tmax(d, c=c):
            base = p['Tmid'] + p['Tamp'] * math.sin(math.pi * d / L)
            bump = p['storm_bump_amp'] * math.exp(-((d - c) ** 2) / (2.0 * p['storm_bump_width'] ** 2))
            return base + bump

        _, exc = gdd_and_flower_exceed(tmax, fs, fe)
        if exc > best_exc:
            best_exc = exc
            best_c = c
    return best_c


def held_out_rows(testId, p):
    fs, fe = flower_window(0.0, p)  # an otherwise-ordinary season
    c_star = storm_center(fs, fe, p)
    w_ho = p['storm_bump_width']
    offs = [-WIDTH_MULT * w_ho + 2 * WIDTH_MULT * w_ho * i / (N_HOLDOUT - 1) for i in range(N_HOLDOUT)]
    rng = random.Random(90000 + testId * 15485863)
    rows = []
    for off in offs:
        center = c_star + off
        day_noise = [rng.gauss(0.0, p['day_noise_std']) for _ in range(L)]

        def tmax(d):
            base = p['Tmid'] + p['Tamp'] * math.sin(math.pi * d / L) + day_noise[d - 1]
            bump = p['storm_bump_amp'] * math.exp(-((d - center) ** 2) / (2.0 * p['storm_bump_width'] ** 2))
            return base + bump

        gdd, flower_exc = gdd_and_flower_exceed(tmax, fs, fe)
        y = p['Y0'] + p['a_gdd'] * gdd - p['BETA'] * (flower_exc ** 2) + rng.gauss(0.0, p['holdout_noise_std'])
        rows.append((gdd, flower_exc, y))
    return rows


# ---------- safe expression parsing ----------
ALLOWED_FUNCS = {
    "abs": abs,
    "exp": lambda x: math.exp(max(-700.0, min(700.0, x))),
    "sin": math.sin,
    "cos": math.cos,
    "sqrt": lambda x: math.sqrt(x) if x >= 0 else float("nan"),
    "tanh": math.tanh,
}
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
)


def parse_expression(text):
    text = text.strip()
    if not text:
        fail("empty output")
    if len(text) > 2000:
        fail("expression too long")
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    nodes = 0
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            fail("disallowed syntax %s" % type(node).__name__)
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant)):
            nodes += 1
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS):
                fail("disallowed call")
            if node.keywords or len(node.args) != 1:
                fail("bad function arity")
        if isinstance(node, ast.Name):
            if node.id in ALLOWED_FUNCS:
                continue
            if node.id not in ("G", "H"):
                fail("unknown name %s" % node.id)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                fail("non-numeric constant")
            v = float(node.value)
            if v != v or v in (float("inf"), float("-inf")):
                fail("non-finite constant")
            if abs(v) > MAX_LITERAL:
                fail("constant out of range")
    if nodes > MAX_NODES:
        fail("expression too large (%d nodes)" % nodes)
    try:
        code = compile(tree, "<expr>", "eval")
    except Exception:
        fail("compile error")
    return code


def eval_expr(code, G, H):
    env = dict(ALLOWED_FUNCS)
    env["G"] = G
    env["H"] = H
    try:
        v = eval(code, {"__builtins__": {}}, env)
    except Exception:
        fail("evaluation error")
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        fail("non-numeric result")
    v = float(v)
    if v != v or v in (float("inf"), float("-inf")):
        fail("non-finite result")
    return v


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            header = fh.readline().split()
        testId = int(header[0])
        n_train_declared = int(header[1])
    except Exception:
        fail("bad instance header")
    if testId < 1 or testId > 100000 or n_train_declared != N_TRAIN:
        fail("bad test id / header")

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace")

    code = parse_expression(text)

    p = instance_params(testId)
    train_rows = regenerate_training(testId, p)
    a0, a1 = ols_g_only(train_rows)
    holdout = held_out_rows(testId, p)

    se = 0.0
    base_se = 0.0
    for G, H, y in holdout:
        pred = eval_expr(code, G, H)
        se += (pred - y) ** 2
        base_pred = a0 + a1 * G
        base_se += (base_pred - y) ** 2

    F = se / len(holdout)
    B = base_se / len(holdout)
    sc = min(CAP, 100.0 * (EPS + B) / (EPS + F))
    print("held_out_MSE=%.6f baseline_MSE=%.6f  Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
