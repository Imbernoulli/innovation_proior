#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the GNSS multipath repeat-period forecasting task.

- Reads the test id AND the number of training rows from <in>'s header, then
  regenerates the hidden law (repeat period P1, per-harmonic amplitudes and
  phases, the exact-24h solar term, and the noise) entirely from the test id --
  the hidden law lives ONLY here and in gen.py (never printed to the solver).
- Regenerates the HELD-OUT grading horizon: a window of `NH` samples starting
  weeks-to-months after the training window ends (a fresh, independent
  pseudo-random gap per test id), with its OWN independent noise draw.
- Parses the participant's closed-form expression over the single variable
  `t` (arithmetic + sin/cos), evaluates it at every held-out `t`, and scores
  by held-out MSE against an internal "predict 0" baseline:
      sc = min(880, 100 * B / max(1e-9, MSE))         B = mean(e_held^2)
      Ratio = sc / 1000
  A constant-0 predictor reproduces B (Ratio ~ 0.1).  The 880 cap keeps the
  ceiling below 0.9 even for a noise-floor-limited perfect period recovery,
  leaving headroom above the "strong" reference.
"""
import sys, math, ast, random

SOLAR_PERIOD = 86400.0
DT = 300.0
NH = 260
MAX_NODES = 100
MAX_OUT_BYTES = 20000
CAP = 880.0  # ceiling of 0.88 -- keeps headroom above even a noise-floor-limited fit

ALLOWED_FUNCS = {
    "sin": math.sin,
    "cos": math.cos,
}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden GNSS multipath law (identical to gen.py) ----------
def hidden_params(t):
    rng = random.Random(910177 + t * 104729)
    gap = rng.uniform(120.0, 420.0)
    P1 = SOLAR_PERIOD - gap
    A0 = rng.uniform(0.12, 0.42)
    phi0 = rng.uniform(0.0, 2 * math.pi)
    harmonics = []
    for lo, hi in [(0.55, 1.05), (0.18, 0.45), (0.05, 0.18)]:
        A = rng.uniform(lo, hi)
        phi = rng.uniform(0.0, 2 * math.pi)
        harmonics.append((A, phi))
    return P1, (A0, phi0), harmonics


def true_signal(tsec, P1, solar, harmonics):
    A0, phi0 = solar
    w0 = 2 * math.pi / SOLAR_PERIOD
    val = A0 * math.cos(w0 * tsec + phi0)
    w1 = 2 * math.pi / P1
    for i, (A, phi) in enumerate(harmonics, start=1):
        val += A * math.cos(i * w1 * tsec + phi)
    return val


def sensor_noise_sigma(t, solar, harmonics):
    """Identical to gen.py's noise model: a per-instance FRACTION `k` of the
    signal's own RMS amplitude, so the noise floor scales with each
    instance's hidden amplitudes instead of being a fixed absolute number."""
    A0, _ = solar
    b = 0.5 * (A0 * A0 + sum(A * A for A, _ in harmonics))
    k_tbl = [0.25, 0.28, 0.30, 0.33, 0.36, 0.40, 0.44, 0.48, 0.55, 0.62]
    k = k_tbl[max(1, min(10, t)) - 1]
    return k * math.sqrt(b)


def heldout_times(t, n_train):
    """Held-out grading horizon: weeks-to-months after the training window ends."""
    rng = random.Random(55555 + t * 7793)
    gap_days = rng.uniform(21.0, 65.0)
    start = n_train * DT + gap_days * 86400.0
    return [start + i * DT for i in range(NH)]


# ---------- expression parsing / validation ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
)


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
            nm = node.id
            if nm in ALLOWED_FUNCS:
                continue
            if nm != "t":
                return "unknown name %s" % nm
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


def compile_expr(raw):
    text = raw.strip()
    if not text:
        fail("empty expression")
    if "\n" in text.strip("\n"):
        # allow trailing newlines but reject multi-statement / multi-line submissions
        text = text.strip()
        if "\n" in text:
            fail("expression must be a single line")
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
    if nodes == 0:
        fail("empty expression")
    try:
        code = compile(tree, "<expr>", "eval")
    except Exception:
        fail("compile error")
    return code, nodes


def eval_expr(code, tsec):
    env = dict(ALLOWED_FUNCS)
    env["t"] = tsec
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
        n_train = int(header[0])
        t = int(header[1])
    except Exception:
        fail("bad instance header")
    if t < 1 or t > 100000 or n_train < 1:
        fail("bad test id")

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace")

    code, nodes = compile_expr(text)

    # regenerate hidden law + held-out horizon (never derived from the submission)
    P1, solar, harmonics = hidden_params(t)
    times = heldout_times(t, n_train)
    rng = random.Random(777791 + t * 15485867)
    sigma = sensor_noise_sigma(t, solar, harmonics)
    truth = [true_signal(ts, P1, solar, harmonics) + rng.gauss(0.0, sigma) for ts in times]

    se = 0.0
    b_acc = 0.0
    for ts, yv in zip(times, truth):
        p = eval_expr(code, ts)
        se += (p - yv) ** 2
        b_acc += yv * yv
    F = se / len(truth)
    B = b_acc / len(truth)

    sc = min(CAP, 100.0 * B / max(1e-9, F))
    print("heldout_MSE=%.6f baseline_MSE=%.6f nodes=%d  Ratio: %.6f"
          % (F, B, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
