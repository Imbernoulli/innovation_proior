#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the learning-curve-forecast task.

- Reads the test id `t` from <in>'s header, then regenerates the SAME hidden
  law used by gen.py (floor, two power-law regimes joined at n_break) --
  identical code, never imported, duplicated here on purpose so no ground
  truth module is ever shipped in the problem directory.
- Regenerates a HELD-OUT set of scales: n_max * {3, 8, 20, 60, 180, 600, 2000}
  where n_max is the largest training n for this instance -- genuine
  extrapolation, 3x-2000x beyond anything the solver has seen, always deep in
  the SLOW asymptotic regime (n_break is always well below n_max by
  construction).
- Parses the participant's ONE-LINE closed-form expression for err(n): a
  Python expression over the variable `n`, operators `+ - * / **`, unary
  minus, numeric constants and the functions sqrt/log/exp/abs. Strictly
  validated via ast (no other names/calls; finite constants only).
- Evaluates the expression at each held-out n (any non-finite result, domain
  error, or overflow anywhere -> the whole submission scores 0).
- Scores from held-out MSE via a log-compressed baseline ratio (prevents the
  huge raw-MSE gap between a floor-blind and a floor-aware fit from
  saturating the score):
      metric   = held-out MSE
      baseline = held-out MSE of the constant predictor mean(train err)
      L(x)     = log1p(max(x, EPS) / EPS)             # EPS fixed, tiny; clamp
                                                       # keeps the ratio bounded
      Ratio    = min(1000, 100 * L(baseline) / L(metric)) / 1000
  The constant predictor reproduces baseline exactly (Ratio == 0.1). Lower
  held-out MSE raises the score; the log compression keeps headroom above a
  good floor-aware fit (measurement noise sets a hard floor on how well the
  floor itself can ever be pinned down from finite data).
"""
import sys, math, ast, random

EPS = 1e-3
MAX_OUT_BYTES = 20000
MAX_EXPR_LEN = 400
MAX_NODES = 40
HELD_OUT_MULT = (3, 8, 20, 60, 180, 600, 2000)


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden learning-curve law (identical to gen.py) ----------
def params(t):
    rng = random.Random(90210 + t * 104729)
    floor = rng.uniform(0.03, 0.11)
    alpha_slow = rng.uniform(0.28, 0.50)
    alpha_fast = alpha_slow + rng.uniform(0.55, 1.05)
    A_slow = rng.uniform(0.8, 2.2)
    n_min = 40 + (t % 4) * 10
    scale_mult = 55.0 + 7.0 * t
    n_max = int(n_min * scale_mult)
    frac = rng.uniform(0.22, 0.38)
    n_break = int(round(n_min * (n_max / n_min) ** frac))
    A_fast = A_slow * (n_break ** (alpha_fast - alpha_slow))
    m = max(9, 14 - (t - 1) // 3)
    sigma = 0.004 + 0.0012 * t
    return floor, A_slow, alpha_slow, A_fast, alpha_fast, n_break, n_min, n_max, m, sigma


def true_err(n, floor, A_slow, alpha_slow, A_fast, alpha_fast, n_break):
    if n < n_break:
        return floor + A_fast * (n ** (-alpha_fast))
    return floor + A_slow * (n ** (-alpha_slow))


def train_ns(n_min, n_max, m):
    ns = []
    prev = 0
    for i in range(m):
        f = i / (m - 1) if m > 1 else 0.0
        n = int(round(n_min * (n_max / n_min) ** f))
        if n <= prev:
            n = prev + 1
        ns.append(n)
        prev = n
    return ns


# ---------- expression parsing / validation ----------
_ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
    "abs": abs,
}
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
)


def _count_nodes(tree):
    return sum(1 for nd in ast.walk(tree)
               if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant)))


def parse_expr(text):
    text = text.strip()
    if not text:
        fail("empty expression")
    if len(text) > MAX_EXPR_LEN:
        fail("expression too long")
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            fail("disallowed syntax %s" % type(node).__name__)
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS):
                fail("disallowed call")
            if node.keywords or len(node.args) != 1:
                fail("bad function arity")
        if isinstance(node, ast.Name):
            if node.id != "n" and node.id not in _ALLOWED_FUNCS:
                fail("unknown name %s" % node.id)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                fail("non-numeric constant")
            v = float(node.value)
            if v != v or v in (float("inf"), float("-inf")):
                fail("non-finite constant")
    nodes = _count_nodes(tree)
    if nodes > MAX_NODES:
        fail("expression too large (%d nodes)" % nodes)
    try:
        code = compile(tree, "<expr>", "eval")
    except Exception:
        fail("compile error")
    return code


def eval_expr(code, n_val):
    env = dict(_ALLOWED_FUNCS)
    env["n"] = float(n_val)
    try:
        v = eval(code, {"__builtins__": {}}, env)
    except Exception:
        fail("evaluation error at n=%s" % n_val)
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        fail("non-numeric result at n=%s" % n_val)
    v = float(v)
    if v != v or v in (float("inf"), float("-inf")):
        fail("non-finite result at n=%s" % n_val)
    return v


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            lines = fh.read().split("\n")
        header = lines[0].split()
        m_train = int(header[0])
        t = int(header[1])
        train_rows = []
        for i in range(1, 1 + m_train):
            a, b = lines[i].split()
            train_rows.append((float(a), float(b)))
    except Exception:
        fail("bad instance file")
    if t < 1 or t > 200000:
        fail("bad test id")

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace").strip()
    # take the first non-empty line only
    first_line = ""
    for ln in text.splitlines():
        if ln.strip():
            first_line = ln.strip()
            break
    code = parse_expr(first_line)

    # regenerate hidden law + held-out scales
    floor, A_slow, alpha_slow, A_fast, alpha_fast, n_break, n_min, n_max, m, sigma = params(t)
    held_ns = [n_max * k for k in HELD_OUT_MULT]

    se = 0.0
    for n in held_ns:
        yt = true_err(n, floor, A_slow, alpha_slow, A_fast, alpha_fast, n_break)
        yp = eval_expr(code, n)
        se += (yp - yt) ** 2
    F_mse = se / len(held_ns)

    # baseline: constant predictor = mean of the TRAIN errors (checker's own
    # trivial feasible construction; ignores decay entirely)
    train_mean = sum(y for _, y in train_rows) / max(1, len(train_rows))
    b_se = 0.0
    for n in held_ns:
        yt = true_err(n, floor, A_slow, alpha_slow, A_fast, alpha_fast, n_break)
        b_se += (train_mean - yt) ** 2
    B_mse = b_se / len(held_ns)

    # clamp both metrics at EPS itself: a submission cannot be rewarded for
    # "beating" the measurement-noise floor by more than a bounded amount,
    # which is what keeps a very lucky/very well-fit case from saturating
    # the ratio cap (irreducible noise really does bound how precisely the
    # floor can ever be pinned down from finitely many noisy rows).
    LF = math.log1p(max(F_mse, EPS) / EPS)
    LB = math.log1p(max(B_mse, EPS) / EPS)
    sc = min(1000.0, 100.0 * LB / max(1e-9, LF))
    print("heldout_MSE=%.8f baseline_MSE=%.8f n_break=%d n_max=%d  Ratio: %.6f"
          % (F_mse, B_mse, n_break, n_max, sc / 1000.0))


if __name__ == "__main__":
    main()
