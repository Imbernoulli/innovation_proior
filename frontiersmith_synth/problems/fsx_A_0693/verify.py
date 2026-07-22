#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the censored call-center hold-time recovery task.

- Reads n_train, test id, and the cap T from <in>'s header (the raw training
  rows themselves are not needed here -- only the count/id, to regenerate the
  same held-out regime deterministically; the hidden law lives ONLY in this
  file + gen.py, never in <in>).
- Regenerates a HELD-OUT set of load values `rho` drawn from a NEAR-CRITICAL
  band (closer to the true, unpublished rho_max than any training load), with
  the same hidden law: mean_hold(rho) = C / (1 - rho/rho_max). The
  held-out target for each rho is this exact analytic mean (no additional
  noise layered on top -- the per-call exponential variance already lives in
  the TRAINING rows the solver saw; here we grade the MEAN FUNCTION itself).
- Parses the participant's submitted expression (arithmetic over `rho`, the
  functions log/sqrt/exp/abs, comparisons, and a ternary `A if cond else B`),
  evaluates it on every held-out rho, and scores held-out MSE with a small
  node-count parsimony penalty (minimisation):
      F = MSE * (1 + LAMBDA * nodes)
      B = MSE_of(train_mean_hold) * (1 + LAMBDA * 1)      # internal baseline
      Ratio = min(920, 100*B/F) / 1000
  A constant predictor (the mean of the CENSORED training holds) reproduces
  the baseline exactly (~0.1). A smooth regression fit directly to the
  censored training holds -- ignoring that a chunk of them are capped at T,
  not really that value -- systematically underestimates the divergence and
  is a trap: it plateaus far below a model that infers rho_max from the
  CENSORING RATE itself. The score is capped below 1.0 on every case so no
  reference can saturate a favourable draw.
"""
import sys, math, ast, random

LAMBDA = 0.005
MAX_NODES = 100
MAX_OUT_BYTES = 20000
M_HELDOUT = 250

ALLOWED_FUNCS = {
    "log": lambda x: math.log(x) if x > 1e-12 else math.log(1e-12),
    "sqrt": lambda x: math.sqrt(x) if x >= 0 else math.sqrt(-x),
    "exp": lambda x: math.exp(max(-60.0, min(60.0, x))),
    "abs": abs,
}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden law (identical to gen.py; never printed to solvers) ----------
RHO_MAX = 0.92
C = 0.55
T_CAP = 3.0


def true_mean_hold(rho):
    return C / (1.0 - rho / RHO_MAX)


def held_out_regime(t):
    band_start = 0.84 * RHO_MAX
    gap_end = max(0.02, 0.10 - 0.007 * (t - 1))
    band_end = RHO_MAX * (1.0 - gap_end)
    if band_end <= band_start:
        band_end = band_start + 0.01
    return band_start, band_end


def gen_heldout(t):
    band_start, band_end = held_out_regime(t)
    rnd = random.Random(70003 + 911 * t)
    rhos, ys = [], []
    for _ in range(M_HELDOUT):
        rho = rnd.uniform(band_start, band_end)
        rhos.append(rho)
        ys.append(true_mean_hold(rho))
    return rhos, ys


# ---------- expression parsing / validation ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.IfExp, ast.Compare, ast.Call,
    ast.Name, ast.Load, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
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
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                return "chained comparisons not allowed"
        if isinstance(node, ast.Name):
            if node.id != "rho" and node.id not in ALLOWED_FUNCS:
                return "disallowed name %s" % node.id
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                return "non-numeric constant"
            v = float(node.value)
            if v != v or v in (float("inf"), float("-inf")):
                return "non-finite constant"
    return None


def _count_nodes(tree):
    return sum(1 for nd in ast.walk(tree)
               if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.IfExp, ast.Compare,
                                   ast.Call, ast.Name, ast.Constant)))


def compile_submission(text):
    text = text.strip()
    if not text:
        fail("empty expression")
    if len(text) > 4000:
        fail("expression too long")
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


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            header = fh.readline().split()
        n_train = int(header[0])
        t = int(header[1])
        t_cap_in = float(header[2])
    except Exception:
        fail("bad instance header")
    if t < 1 or t > 100000 or n_train < 1:
        fail("bad test id / n_train")
    if abs(t_cap_in - T_CAP) > 1e-6:
        fail("instance cap mismatch")

    try:
        train_holds = []
        with open(inf) as fh:
            fh.readline()
            for _ in range(n_train):
                parts = fh.readline().split()
                train_holds.append(float(parts[1]))
    except Exception:
        fail("cannot read training rows")

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace")

    code, nodes = compile_submission(text)

    rhos, ys = gen_heldout(t)

    glob = {"__builtins__": {}}
    se = 0.0
    for rho, y in zip(rhos, ys):
        env = dict(ALLOWED_FUNCS)
        env["rho"] = rho
        try:
            p = eval(code, glob, env)
        except Exception:
            fail("evaluation error")
        if not isinstance(p, (int, float)) or isinstance(p, bool):
            fail("non-numeric result")
        p = float(p)
        if p != p or p in (float("inf"), float("-inf")):
            fail("non-finite result")
        se += (p - y) ** 2
    F_mse = se / len(ys)

    train_mean = sum(train_holds) / len(train_holds)
    B_mse = sum((train_mean - y) ** 2 for y in ys) / len(ys)

    Fscore = F_mse * (1.0 + LAMBDA * nodes)
    B = B_mse * (1.0 + LAMBDA * 1)
    # Cap below 1000 (i.e. Ratio < 0.92) on EVERY case so no reference solution
    # can saturate a favourable draw to a perfect score.
    sc = min(920.0, 100.0 * B / max(1e-9, Fscore))
    print("heldout_MSE=%.6f baseline_MSE=%.6f nodes=%d  Ratio: %.6f"
          % (F_mse, B_mse, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
