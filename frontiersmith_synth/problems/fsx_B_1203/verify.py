#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the tide-plus-surge held-out storm forecast.

- Reads testId from <in>'s header, then regenerates -- entirely from testId,
  identically to gen.py -- the hidden harmonic-tide constituents, the calm
  surge driver, the shallow-water interaction coefficient kappa, and a
  HELD-OUT STORM EVENT (never shown to the solver): a surge pulse timed to
  peak near a HIGH tide (an adversarial "peak-on-peak" trap for anyone who
  just adds tide and surge).
- Parses the participant's output: ONE arithmetic expression string over
  the two variables T (tide) and S (surge proxy), using +, -, *, /, **,
  parentheses, numeric constants and a small whitelist of unary math
  functions.  No conditionals, comparisons, names other than T/S/funcs, or
  any other syntax are allowed.
- Evaluates the expression at every held-out (T, S) row (rolling state is
  NOT needed here -- this is a plain scalar predictor), rejecting any
  non-finite value.
- Scores by held-out MSE against an internal baseline predictor
  (pred = T: "assume it's an ordinary tide day, ignore the surge
  entirely") -- the naive status-quo forecast the theme is named for:
      F = held-out MSE of the submission
      B = held-out MSE of the baseline (T only)
      Ratio = min(920, 100*(eps+B)/(eps+F)) / 1000
  A predictor that reproduces the baseline scores ~0.1.  One that folds in
  the surge (but not the interaction) does much better.  One that also
  applies the given interaction correction does best -- but the ratio is
  hard-capped at 0.92 so even a perfect fit (bounded by storm sensor
  noise) leaves headroom above the reference solution.
"""
import sys, math, ast

FREQS = [1.0, 1.9322, 0.1341]
N_TRAIN = 150
T_SPAN = 30.0
N_HOLDOUT = 40
WIDTH_MULT = 1.6
STORM_LO, STORM_HI = 30.0, 60.0
STORM_GRID = 3000
EPS = 1e-4
CAP = 920.0
MAX_OUT_BYTES = 20000
MAX_NODES = 60
MAX_LITERAL = 1e6


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden station parameters (identical to gen.py) ----------
def instance_params(testId):
    import random
    rng = random.Random(20000 + testId * 7919)
    amps = [0.7 + 0.3 * rng.random() for _ in FREQS]
    phases = [rng.uniform(0.0, 2 * math.pi) for _ in FREQS]
    T0 = 0.0
    surge_freq = 0.31 + 0.01 * rng.random()
    surge_phase = rng.uniform(0.0, 2 * math.pi)
    calm_surge_amp = 0.08 + 0.06 * rng.random()
    kappa = 0.30 + 0.02 * testId + 0.003 * rng.random()
    storm_mult = 2.5 + 0.3 * testId + 0.09 * rng.random()
    storm_width = 1.2 + 0.1 * rng.random()
    train_noise_std = 0.02
    storm_noise_std = 0.10
    return dict(amps=amps, phases=phases, T0=T0, surge_freq=surge_freq,
                surge_phase=surge_phase, calm_surge_amp=calm_surge_amp,
                kappa=kappa, storm_mult=storm_mult, storm_width=storm_width,
                train_noise_std=train_noise_std, storm_noise_std=storm_noise_std)


def tide(t, p):
    return p['T0'] + sum(A * math.cos(2 * math.pi * f * t + ph)
                          for A, f, ph in zip(p['amps'], FREQS, p['phases']))


def calm_surge(t, p):
    return p['calm_surge_amp'] * math.sin(2 * math.pi * p['surge_freq'] * t + p['surge_phase'])


def storm_center(p):
    """Deterministic grid search for the high-tide time nearest the storm window --
    no RNG, pure arithmetic, so it matches bit-for-bit between calls."""
    best_t, best_T = STORM_LO, -1e18
    for i in range(STORM_GRID):
        t = STORM_LO + (STORM_HI - STORM_LO) * i / (STORM_GRID - 1)
        Tv = tide(t, p)
        if Tv > best_T:
            best_T = Tv
            best_t = t
    return best_t


def held_out_rows(testId, p):
    import random
    tc = storm_center(p)
    w = p['storm_width']
    offs = [-WIDTH_MULT * w + 2 * WIDTH_MULT * w * i / (N_HOLDOUT - 1) for i in range(N_HOLDOUT)]
    rng = random.Random(90000 + testId * 15485863)
    rows = []
    for off in offs:
        t = tc + off
        T = tide(t, p)
        Sc = calm_surge(t, p)
        bump = p['storm_mult'] * p['calm_surge_amp'] * math.exp(-(off ** 2) / (2 * w ** 2))
        S = Sc + bump
        noise = rng.gauss(0.0, p['storm_noise_std'])
        y = T + S - p['kappa'] * T * S + noise
        rows.append((T, S, y))
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
            if node.id not in ("T", "S"):
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


def eval_expr(code, T, S):
    env = dict(ALLOWED_FUNCS)
    env["T"] = T
    env["S"] = S
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
    holdout = held_out_rows(testId, p)

    se = 0.0
    base_se = 0.0
    for T, S, y in holdout:
        pred = eval_expr(code, T, S)
        se += (pred - y) ** 2
        base_se += (T - y) ** 2

    F = se / len(holdout)
    B = base_se / len(holdout)
    sc = min(CAP, 100.0 * (EPS + B) / (EPS + F))
    print("held_out_MSE=%.6f baseline_MSE=%.6f  Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
