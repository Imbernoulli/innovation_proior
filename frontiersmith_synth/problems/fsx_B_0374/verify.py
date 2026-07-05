#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the wind-tunnel sensor scaling-law extrapolation task.

- Reads the test id from <in> (first line), then regenerates the ground-truth
  scaling law, the TRAIN sample (for the baseline mean) and the HELD-OUT
  EXTRAPOLATION split (LARGE compute / resolution) entirely from that id.  The
  law lives ONLY here.
- Parses the participant's closed-form expression <out> over {x1=C, x2=R}
  through a strict AST whitelist (rejects imports/attributes/unknown names,
  non-finite results, oversized input).
- Score (minimisation, complexity-penalised held-out MSE):
      F = heldout_MSE * (1 + LAMBDA * complexity)
      B = baseline_MSE * (1 + LAMBDA * 1)      # baseline = constant train mean
      Ratio = min(1000, 100*B/F) / 1000
  A constant reproduces the baseline (~0.1).  A pure power law (no floor)
  extrapolates poorly to large C,R because L there is dominated by the
  irreducible floor; identifying the floor drives held-out error toward the
  irreducible-noise level and raises the ratio, but that noise keeps it below 1.
"""
import sys, math, ast, random

LAMBDA = 0.003
# held-out EXTRAPOLATION regime: much larger compute / resolution than train
C_HELD_LO, C_HELD_HI = 25.0, 160.0
R_HELD_LO, R_HELD_HI = 25.0, 160.0
N_HELD = 400
GAMMA = 0.30      # signal-proportional part of irreducible held-out noise
DELTA = 0.14      # baseline-proportional part (guarantees anti-saturation headroom)
ALLOWED_FUNCS = {"exp": math.exp, "log": math.log, "sin": math.sin,
                 "cos": math.cos, "sqrt": math.sqrt, "tanh": math.tanh,
                 "abs": abs, "pow": pow}
ALLOWED_VARS = {"x1", "x2"}
MAX_EXPR_BYTES = 200000


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---- ground truth (identical to gen.py) ----
def coeffs(t):
    rng = random.Random(310007 + t * 6421)
    return (rng.uniform(0.50, 1.20), rng.uniform(1.5, 3.5), rng.uniform(0.35, 0.65),
            rng.uniform(1.2, 3.0), rng.uniform(0.35, 0.65))


def fval(C, R, cf):
    Einf, A, alpha, B, beta = cf
    return Einf + A * C ** (-alpha) + B * R ** (-beta)


C_LO, C_HI = 1.0, 25.0
R_LO, R_HI = 1.0, 25.0


def gen_train(t):
    sigma = 0.04 + (t - 1) * 0.015
    n = 180 - (t - 1) * 12
    cf = coeffs(t)
    rng = random.Random(880 + t * 97711)
    lgC = math.log(C_LO), math.log(C_HI)
    lgR = math.log(R_LO), math.log(R_HI)
    rows = []
    for _ in range(n):
        C = math.exp(rng.uniform(*lgC))
        R = math.exp(rng.uniform(*lgR))
        rows.append(((C, R), fval(C, R, cf) + rng.gauss(0.0, sigma)))
    return rows, cf


def gen_held(t, tm):
    """Extrapolation split.  Irreducible held-out noise has two deterministic
    parts: a signal-proportional term (GAMMA * mean true signal) and a
    baseline-proportional term (DELTA * noiseless baseline MSE).  The latter
    guarantees the best possible fit still cannot beat the constant baseline by
    more than ~1/DELTA, keeping the score away from the saturation cap."""
    cf = coeffs(t)
    rng = random.Random(4241 + t * 20261)
    lgC = math.log(C_HELD_LO), math.log(C_HELD_HI)
    lgR = math.log(R_HELD_LO), math.log(R_HELD_HI)
    pts = []
    for _ in range(N_HELD):
        C = math.exp(rng.uniform(*lgC))
        R = math.exp(rng.uniform(*lgR))
        pts.append([(C, R), fval(C, R, cf)])
    mean_true = sum(p[1] for p in pts) / len(pts)
    B0 = sum((p[1] - tm) ** 2 for p in pts) / len(pts)   # noiseless baseline MSE
    sh = math.sqrt((GAMMA * mean_true) ** 2 + DELTA * B0)
    nrng = random.Random(9001 + t * 15485863)
    for p in pts:
        p[1] = p[1] + nrng.gauss(0.0, sh)
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
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            return "non-numeric constant"
    return None


def complexity(tree):
    return sum(1 for nd in ast.walk(tree)
               if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant)))


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

    train, _ = gen_train(t)
    tm = sum(y for _, y in train) / len(train)
    held = gen_held(t, tm)

    cx = complexity(tree)

    se = 0.0
    for (C, R), yv in held:
        env = {"x1": C, "x2": R}
        env.update(ALLOWED_FUNCS)
        try:
            p = eval(code, {"__builtins__": {}}, env)
        except Exception:
            fail("evaluation error")
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            fail("non-numeric result")
        p = float(p)
        if p != p or p in (float("inf"), float("-inf")):
            fail("non-finite result")
        d = p - yv
        se += d * d
    F_mse = se / len(held)

    B_mse = sum((yv - tm) ** 2 for _, yv in held) / len(held)
    B = B_mse * (1.0 + LAMBDA * 1)
    F = F_mse * (1.0 + LAMBDA * cx)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("heldout_MSE=%.6f baseline_MSE=%.6f complexity=%d  Ratio: %.6f"
          % (F_mse, B_mse, cx, sc / 1000.0))


if __name__ == "__main__":
    main()
