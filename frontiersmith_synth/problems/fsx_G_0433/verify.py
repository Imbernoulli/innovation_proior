#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the planetary-ellipse implicit-conic recovery task.

- Reads the test id from <in> (first line), then regenerates the hidden orbit,
  the TRAIN point cloud and the WITHHELD-ARC extrapolation split entirely from
  that id (the orbit lives ONLY here).
- Parses the participant's implicit relation F(x, y) <out> through a strict AST
  whitelist over {x, y} (rejects imports / attributes / unknown names /
  non-finite constants and results / oversized input).
- The curve is judged by the scale-invariant TAUBIN distance
      d(p) = |F(p)| / ||grad F(p)||         (first-order geometric distance
                                             from p to the zero set F=0)
  averaged over the withheld-arc points, penalised by expression complexity:
      F_obj = mean_d * (1 + LAMBDA * complexity)
  A model whose gradient is numerically degenerate (F ~ constant, e.g. the
  trivial "F = 0" cheat) is rejected.
- Baseline B = the same functional of the grader's own trivial construction:
  the best-fit CIRCLE through the training centroid.  A circle roughly follows
  the orbit but cannot bend to the eccentric withheld arc, so it scores ~0.1.
      Ratio = min(1000, 100 * B / F_obj) / 1000
  Recovering the focus-conic structure (r, x, y linear form) drives the arc
  distance toward the irreducible positional-noise floor and raises the ratio,
  but that noise floor keeps it well below 1.0.  Higher is better.
"""
import sys
import math
import ast
import random

TWO_PI = 2.0 * math.pi
LAMBDA = 0.010
N_HELD = 200
GFLOOR = 1e-6          # gradient-norm floor: below this the curve is degenerate
MAX_EXPR_BYTES = 200000

ALLOWED_FUNCS = {"exp": math.exp, "log": math.log, "sin": math.sin,
                 "cos": math.cos, "sqrt": math.sqrt, "tanh": math.tanh,
                 "abs": abs}
ALLOWED_VARS = {"x", "y"}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---- hidden ground truth (identical to gen.py) ----
def orbit_params(t):
    rng = random.Random(920177 + t * 5273)
    a = rng.uniform(1.6, 3.4)
    e = rng.uniform(0.42, 0.56)
    omega = rng.uniform(0.0, TWO_PI)
    p = a * (1.0 - e * e)
    gwid = rng.uniform(1.0, 1.35)
    glo = rng.uniform(0.35, TWO_PI - gwid - 0.35)
    return a, e, omega, p, glo, glo + gwid


def point(phi, pm):
    a, e, omega, p, glo, ghi = pm
    r = p / (1.0 + e * math.cos(phi - omega))
    return r * math.cos(phi), r * math.sin(phi)


def gen_train(t):
    a, e, omega, p, glo, ghi = pm = orbit_params(t)
    sigma = a * (0.008 + (t - 1) * 0.0012)
    n = 170 - (t - 1) * 7
    rng = random.Random(6600 + t * 74521)
    rows = []
    while len(rows) < n:
        phi = rng.uniform(0.0, TWO_PI)
        if glo <= phi <= ghi:
            continue
        x, y = point(phi, pm)
        x += rng.gauss(0.0, sigma)
        y += rng.gauss(0.0, sigma)
        rows.append((x, y))
    return rows, pm


def gen_held(t):
    """Withheld-arc points on the true orbit + a fixed irreducible positional
    noise floor (independent of the participant)."""
    a, e, omega, p, glo, ghi = pm = orbit_params(t)
    rng = random.Random(410009 + t * 61879)
    floor = a * 0.068
    pts = []
    for _ in range(N_HELD):
        phi = rng.uniform(glo, ghi)
        x, y = point(phi, pm)
        x += rng.gauss(0.0, floor)
        y += rng.gauss(0.0, floor)
        pts.append((x, y))
    return pts


# ---- strict expression validation ----
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.USub, ast.UAdd,
)


def validate_ast(tree):
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return "disallowed syntax %s" % type(node).__name__
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS):
                return "disallowed call"
            if node.keywords:
                return "kwargs not allowed"
        if isinstance(node, ast.Name):
            if node.id not in ALLOWED_VARS and node.id not in ALLOWED_FUNCS:
                return "unknown name %s" % node.id
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                return "non-numeric constant"
            if not math.isfinite(float(node.value)):
                return "non-finite constant"
    return None


def complexity(tree):
    return sum(1 for nd in ast.walk(tree)
               if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant)))


def make_eval(code):
    def fn(x, y):
        env = {"x": x, "y": y}
        env.update(ALLOWED_FUNCS)
        v = eval(code, {"__builtins__": {}}, env)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("non-numeric")
        v = float(v)
        if not math.isfinite(v):
            raise ValueError("non-finite")
        return v
    return fn


def taubin_mean(fn, pts):
    """Mean first-order (Taubin) distance from held-out points to F=0.
    Returns None if the curve is numerically degenerate at some point."""
    tot = 0.0
    for x, y in pts:
        h = 1e-5 * (1.0 + abs(x) + abs(y))
        f0 = fn(x, y)
        gx = (fn(x + h, y) - fn(x - h, y)) / (2.0 * h)
        gy = (fn(x, y + h) - fn(x, y - h)) / (2.0 * h)
        gn = math.sqrt(gx * gx + gy * gy)
        if gn < GFLOOR:
            return None
        tot += abs(f0) / gn
    return tot / len(pts)


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
    if t < 1 or t > 10000:
        fail("bad test id")

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_EXPR_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_EXPR_BYTES:
        fail("output too large")
    expr = raw.decode("utf-8", "replace").strip()
    if not expr:
        fail("empty expression")
    lines = [ln for ln in expr.splitlines() if ln.strip()]
    if len(lines) != 1:
        fail("expression must be a single line")
    expr = lines[0].strip()

    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        fail("parse error")
    reason = validate_ast(tree)
    if reason:
        fail(reason)
    try:
        code = compile(tree, "<expr>", "eval")
    except Exception:
        fail("compile error")

    held = gen_held(t)
    fn = make_eval(code)
    try:
        md = taubin_mean(fn, held)
    except Exception:
        fail("evaluation error")
    if md is None:
        fail("degenerate gradient (curve is numerically constant)")

    cx = complexity(tree)
    F = md * (1.0 + LAMBDA * cx)

    # ---- grader's own baseline: best-fit circle through training centroid ----
    train, _ = gen_train(t)
    cxm = sum(p[0] for p in train) / len(train)
    cym = sum(p[1] for p in train) / len(train)
    R = sum(math.hypot(p[0] - cxm, p[1] - cym) for p in train) / len(train)
    base_expr = "(x - (%r))**2 + (y - (%r))**2 - (%r)" % (cxm, cym, R * R)
    bcode = compile(ast.parse(base_expr, mode="eval"), "<b>", "eval")
    bfn = make_eval(bcode)
    bmd = taubin_mean(bfn, held)
    bcx = complexity(ast.parse(base_expr, mode="eval"))
    B = bmd * (1.0 + LAMBDA * bcx)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("arc_taubin=%.6f baseline_taubin=%.6f complexity=%d  Ratio: %.6f"
          % (md, bmd, cx, sc / 1000.0))


if __name__ == "__main__":
    main()
