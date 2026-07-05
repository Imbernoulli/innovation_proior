#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the orbital-mechanics conserved-quantity task.

- Reads the test id from <in> (first line).  Regenerates -- entirely from that id
  and seeds baked in HERE -- the hidden gravitational parameter mu and a set of
  HELD-OUT GRADING ORBITS.  The grading orbits are LARGER than the training orbits
  the solver saw (a genuine extrapolation region); the ground truth lives ONLY in
  this file.
- Parses the participant's closed-form expression C(x1,x2,x3,x4) through a strict
  AST whitelist (rejects imports/attributes/unknown names, non-finite results,
  oversized input).
- A quantity is "conserved" if it is nearly constant WITHIN each orbit while still
  differing BETWEEN orbits.  We score with the correlation ratio (eta^2), the
  fraction of the pooled variance explained by the between-orbit grouping:
        eta2 = SS_between / SS_total     in [0,1]
  A pointwise expression that is constant along every trajectory is a first
  integral of the motion, so eta2 -> 1 (bounded below 1 by measurement noise).
  A truly constant expression (SS_total = 0) is degenerate and scores 0.

- Score (maximisation, complexity-penalised):
        F = eta2_participant / (1 + LAMBDA*complexity)
        B = eta2_baseline    / (1 + LAMBDA*complexity_baseline)   # baseline = speed^2
        Ratio = min(1000, 100*F/B) / 1000
  The internal baseline is v^2 = x3^2 + x4^2, which is only PARTIALLY conserved
  (v^2 = 2E + 2 mu/r couples the conserved energy to the non-conserved 1/r term),
  so reproducing it scores ~0.1 while a genuine invariant scores far higher; the
  noise floor keeps even the best invariant below 1.0.
"""
import sys, math, ast, random

LAMBDA = 0.006
BASELINE_EXPR = "x3**2 + x4**2"
J_HELD = 20
K = 16
HA_LO, HA_HI = 2.2, 3.6          # held-out orbits are LARGER than train ([1.0,2.0])
HE_LO, HE_HI = 0.1, 0.6
ALLOWED_FUNCS = {"exp": math.exp, "log": math.log, "sin": math.sin,
                 "cos": math.cos, "sqrt": math.sqrt, "tanh": math.tanh,
                 "atan": math.atan, "abs": abs}
ALLOWED_VARS = {"x1", "x2", "x3", "x4"}
MAX_EXPR_BYTES = 200000


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---- hidden ground-truth physics (identical shape to gen.py) ----
def mu_of(t):
    rng = random.Random(90001 + t * 7919)
    return rng.uniform(0.8, 1.5)


def noise_level(t):
    return 0.030 + (t - 1) * 0.005


def kepler_states(a, e, mu, omega, E_list):
    n = math.sqrt(mu / (a ** 3))
    b = math.sqrt(1.0 - e * e)
    co, so = math.cos(omega), math.sin(omega)
    out = []
    for E in E_list:
        cE, sE = math.cos(E), math.sin(E)
        xo = a * (cE - e)
        yo = a * b * sE
        Edot = n / (1.0 - e * cE)
        vxo = -a * sE * Edot
        vyo = a * b * cE * Edot
        x = co * xo - so * yo
        y = so * xo + co * yo
        vx = co * vxo - so * vyo
        vy = so * vxo + co * vyo
        out.append((x, y, vx, vy))
    return out


def gen_held(t):
    mu = mu_of(t)
    nl = noise_level(t)
    rng = random.Random(777 + t * 20261)
    trajs = []
    for _ in range(J_HELD):
        a = rng.uniform(HA_LO, HA_HI)
        e = rng.uniform(HE_LO, HE_HI)
        omega = rng.uniform(0.0, 2.0 * math.pi)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        E_list = [phase + 2.0 * math.pi * k / K + rng.uniform(-0.05, 0.05)
                  for k in range(K)]
        clean = kepler_states(a, e, mu, omega, E_list)
        sp = nl * a
        sv = nl * math.sqrt(mu / a)
        noisy = [(x + rng.gauss(0, sp), y + rng.gauss(0, sp),
                  vx + rng.gauss(0, sv), vy + rng.gauss(0, sv))
                 for (x, y, vx, vy) in clean]
        trajs.append(noisy)
    return trajs


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


def compile_expr(expr):
    tree = ast.parse(expr, mode="eval")
    reason = validate_ast(tree)
    if reason:
        raise ValueError(reason)
    return compile(tree, "<expr>", "eval"), complexity(tree)


def eta2(trajs, code):
    """Correlation ratio of the compiled expression over grouped trajectories.
    Returns (eta2, ok)."""
    vals = []
    groups = []
    for tr in trajs:
        gv = []
        for (x1, x2, x3, x4) in tr:
            env = {"x1": x1, "x2": x2, "x3": x3, "x4": x4}
            env.update(ALLOWED_FUNCS)
            try:
                p = eval(code, {"__builtins__": {}}, env)
            except Exception:
                return 0.0, False
            if not isinstance(p, (int, float)) or isinstance(p, bool):
                return 0.0, False
            p = float(p)
            if p != p or p in (float("inf"), float("-inf")):
                return 0.0, False
            gv.append(p)
            vals.append(p)
        groups.append(gv)
    gm = sum(vals) / len(vals)
    sst = sum((v - gm) ** 2 for v in vals)
    if sst < 1e-12:                      # constant expression -> degenerate
        return 0.0, True
    ssb = 0.0
    for gv in groups:
        m = sum(gv) / len(gv)
        ssb += len(gv) * (m - gm) ** 2
    e2 = ssb / sst
    if e2 < 0.0:
        e2 = 0.0
    if e2 > 1.0:
        e2 = 1.0
    return e2, True


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            header = fh.readline().split()
        t = int(header[2])
    except Exception:
        fail("bad instance header")
    if t < 1 or t > 100000:
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
        code, cx = compile_expr(expr)
    except Exception as ex:
        fail("bad expression: %s" % ex)

    held = gen_held(t)

    e2, ok = eta2(held, code)
    if not ok:
        fail("evaluation error / non-finite result")

    bcode, bcx = compile_expr(BASELINE_EXPR)
    b2, _ = eta2(held, bcode)

    F = e2 / (1.0 + LAMBDA * cx)
    B = b2 / (1.0 + LAMBDA * bcx)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("eta2=%.6f baseline_eta2=%.6f complexity=%d  Ratio: %.6f"
          % (e2, b2, cx, sc / 1000.0))


if __name__ == "__main__":
    main()
