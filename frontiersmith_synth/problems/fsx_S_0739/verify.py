#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for "one physical law hiding behind six lab setups"
(shared-law-nuisance-factoring).

- Reads the test id `t` from <in>'s header only (the K training regimes'
  rows are for the SOLVER, not needed here).
- Regenerates the HIDDEN shared law  core(x) = sin(w*x+phi) + c*x  (same
  seeded function as gen.py) and, ONLY here, a brand-new K+1-th regime that
  was never shown to the solver: its own unknown (gain,offset), a handful of
  CALIBRATION points (x,y) drawn from a narrow window, and a set of
  EXTRAPOLATION points drawn from an x range that lies completely outside
  every training window (and outside the calibration window).
- Parses the participant's submitted CLOSED-FORM expression for core(x)
  (arithmetic over `x`, + - * /, parentheses, numeric constants, and the
  unary functions sin, cos, exp, abs).
- Uses the *calibration* points ONLY to fit a 2-parameter affine map
      y ~= gain * core_hat(x) + offset      (ordinary least squares)
  -- exactly the ratio/difference-style nuisance removal the statement
  describes -- then scores the fitted map's predictions on the *extrapolation*
  points (a genuinely unseen regime AND unseen x territory) via MSE.
      F = heldout_MSE
      B = MSE of the naive "flat calibration-mean" predictor (gain=0)
      Ratio = min(1000, 100*B/F) / 1000
  A shape that is not really core(x) still gets its own best-fit gain/offset
  (so a merely-plausible-looking curve is not unfairly punished for scale),
  but only the TRUE shared shape extrapolates well past the calibration
  window into fresh x territory and a fresh regime.
"""
import sys, math, ast

MAX_NODES = 60
MAX_OUT_BYTES = 20000
AMP = 1.0

ALLOWED_FUNCS = {
    "sin": math.sin,
    "cos": math.cos,
    "exp": lambda v: math.exp(v) if -700.0 < v < 700.0 else (_ for _ in ()).throw(OverflowError()),
    "abs": abs,
}

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
)


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden shared law (identical to gen.py) ----------
def hidden_law(t):
    rng = __import__("random").Random(9001 + t * 7919)
    w = rng.uniform(0.9, 1.7)
    phi = rng.uniform(0.0, 2 * math.pi)
    c = rng.uniform(0.04, 0.18)
    return w, phi, c


def true_core(x, w, phi, c):
    return math.sin(w * x + phi) + c * x


# ---------- N_NEW held-out K+1..K+N_NEW regimes (checker-only, never in gen.py) ----------
# Scoring one single random held-out regime made the Ratio swing wildly from
# test id to test id (whichever regime happened to get an easy/hard
# gain-offset-phase draw). Averaging over several independent held-out
# regimes per test id keeps the SAME mechanism (unseen regime + unseen x)
# while making the score a stable estimate of extrapolation quality rather
# than a single noisy draw.
N_NEW = 4


def new_regime(t, j):
    rng = __import__("random").Random(70021 + t * 15485863 + j * 992299)
    gain = rng.uniform(0.5, 2.2)
    offset = rng.uniform(-4.0, 4.0)
    calib_center = rng.uniform(-6.5, 6.5)
    calib_hw = 1.4  # wide enough (> quarter period) to see real curvature, not a flat spot
    side = rng.choice([-1.0, 1.0])
    extrap_start = 9.75 + 0.08 * (t - 1)
    extrap_width = 1.2
    return gain, offset, calib_center, calib_hw, side, extrap_start, extrap_width


def build_heldout(t):
    """Returns a list of N_NEW (calib_xs, calib_ys, extrap_xs, extrap_ys) tuples."""
    w, phi, c = hidden_law(t)
    sigma = 0.22 + 0.020 * (t - 1)
    out = []
    for j in range(N_NEW):
        gain, offset, calib_center, calib_hw, side, extrap_start, extrap_width = new_regime(t, j)

        rng_c = __import__("random").Random(881001 + t * 733 + j * 611953)
        n_calib = 16
        calib_xs = sorted(rng_c.uniform(calib_center - calib_hw, calib_center + calib_hw) for _ in range(n_calib))
        calib_ys = [gain * true_core(x, w, phi, c) + offset + rng_c.gauss(0.0, sigma) for x in calib_xs]

        rng_e = __import__("random").Random(447701 + t * 2003 + j * 322937)
        n_extrap = 10
        raw = sorted(rng_e.uniform(extrap_start, extrap_start + extrap_width) for _ in range(n_extrap))
        extrap_xs = [side * v for v in raw]
        extrap_ys = [gain * true_core(x, w, phi, c) + offset + rng_e.gauss(0.0, sigma) for x in extrap_xs]

        out.append((calib_xs, calib_ys, extrap_xs, extrap_ys))
    return out


# ---------- expression parsing / validation ----------
def _validate_ast(tree):
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
            if node.id != "x":
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
    text = raw.strip()
    if not text:
        fail("empty output")
    # only look at the first non-empty line
    text = text.splitlines()[0].strip()
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    err = _validate_ast(tree)
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


def eval_core(code, xs):
    glob = {"__builtins__": {}}
    out = []
    for x in xs:
        env = dict(ALLOWED_FUNCS)
        env["x"] = x
        try:
            v = eval(code, glob, env)
        except Exception:
            fail("evaluation error")
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            fail("non-numeric result")
        v = float(v)
        if v != v or v in (float("inf"), float("-inf")):
            fail("non-finite result")
        out.append(v)
    return out


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

    heldout = build_heldout(t)

    # Ridge-regularised gain: a submitted core with almost no variation over
    # the calibration window (e.g. a near-flat line) would otherwise let
    # ordinary least squares divide by a near-zero sxx and amplify pure
    # calibration noise into an enormous, meaningless gain. RIDGE is small
    # relative to a genuinely curved shape's calibration variance, so it
    # barely perturbs a well-conditioned fit but caps the degenerate case.
    RIDGE = 0.5

    se_F = 0.0
    se_B = 0.0
    n_total = 0
    for calib_xs, calib_ys, extrap_xs, extrap_ys in heldout:
        core_calib = eval_core(code, calib_xs)
        core_extrap = eval_core(code, extrap_xs)

        n = len(calib_xs)
        mean_c = sum(core_calib) / n
        mean_y = sum(calib_ys) / n
        sxy = sum((c - mean_c) * (y - mean_y) for c, y in zip(core_calib, calib_ys))
        sxx = sum((c - mean_c) ** 2 for c in core_calib)
        gain = sxy / (sxx + RIDGE)
        offset = mean_y - gain * mean_c

        preds = [gain * c + offset for c in core_extrap]
        for p in preds:
            if p != p or p in (float("inf"), float("-inf")):
                fail("non-finite prediction")

        se_F += sum((p - y) ** 2 for p, y in zip(preds, extrap_ys))
        se_B += sum((mean_y - y) ** 2 for y in extrap_ys)
        n_total += len(extrap_ys)

    F = se_F / n_total
    B = se_B / n_total

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("heldout_MSE=%.6f baseline_MSE=%.6f nodes=%d n_regimes=%d  Ratio: %.6f"
          % (F, B, nodes, len(heldout), sc / 1000.0))


if __name__ == "__main__":
    main()
