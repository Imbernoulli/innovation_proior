#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the sled stick-slip pull-response recovery task.

- Reads n_train and the test id from <in>'s header (data rows themselves are
  ignored here -- only their COUNT/id matter to regenerate the same held-out
  regime; the hidden law lives ONLY in this file + gen.py, never in <in>).
- Regenerates a HELD-OUT set of (F, r) pull attempts under a WIDER (more
  aggressive force, longer rest) regime than training, with the same hidden
  law: an aging static-friction guard  hold(r) = F0 + A*ln(1+r)  picks between
  a logarithmic creep law (stuck) and a linear excess-force law (sliding).
- Parses the participant's submitted expression (arithmetic over F, r, the
  functions log/sqrt/exp/abs, comparisons, and a ternary `A if cond else B`),
  evaluates it on every held-out (F, r), and scores held-out MSE with a small
  node-count parsimony penalty (minimisation):
      F = MSE * (1 + LAMBDA * nodes)
      B = MSE_of(train_mean_y) * (1 + LAMBDA * 1)      # internal baseline
      Ratio = min(1000, 100*B/F) / 1000
  A constant predictor reproduces the baseline exactly (~0.1). A single smooth
  formula (no switching) cannot capture the discontinuous jump at the guard,
  so it plateaus well below a model that recovers the guard. Noise + a wider
  extrapolation window at high test ids keep even a good model off the
  ceiling.
"""
import sys, math, ast, random

LAMBDA = 0.005
MAX_NODES = 100
MAX_OUT_BYTES = 20000
M_HELDOUT = 300

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
F0, A, FK, KC, AKIN, BKIN = 3.0, 1.1, 1.2, 0.04, 0.9, 0.25


def true_y(F, r):
    L = math.log(1.0 + r)
    hold = F0 + A * L
    if F <= hold:
        return KC * F * L
    else:
        return AKIN * (F - FK) + BKIN


def held_out_regime(t):
    sigma_add = 0.03
    sigma_rel = 0.02 + 0.02 * (t - 1)
    f_max = 7.0 + 0.5 * (t - 1)
    r_max = 15.0 + 2.0 * (t - 1)
    return sigma_add, sigma_rel, f_max, r_max


def gen_heldout(t):
    sigma_add, sigma_rel, f_max, r_max = held_out_regime(t)
    rnd = random.Random(31337 + 977 * t)
    Fs, rs, ys = [], [], []
    for _ in range(M_HELDOUT):
        F = rnd.uniform(0.0, f_max)
        r = rnd.uniform(0.0, r_max)
        yt = true_y(F, r)
        noise_sd = sigma_add + sigma_rel * abs(yt)
        y = yt + rnd.gauss(0.0, noise_sd)
        Fs.append(F); rs.append(r); ys.append(y)
    return Fs, rs, ys


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
            if node.id not in ("F", "r") and node.id not in ALLOWED_FUNCS:
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
    except Exception:
        fail("bad instance header")
    if t < 1 or t > 100000 or n_train < 1:
        fail("bad test id / n_train")

    try:
        train_ys = []
        with open(inf) as fh:
            fh.readline()
            for _ in range(n_train):
                parts = fh.readline().split()
                train_ys.append(float(parts[2]))
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

    Fs, rs, ys = gen_heldout(t)

    glob = {"__builtins__": {}}
    se = 0.0
    for F, r, y in zip(Fs, rs, ys):
        env = dict(ALLOWED_FUNCS)
        env["F"] = F
        env["r"] = r
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

    train_mean = sum(train_ys) / len(train_ys)
    B_mse = sum((train_mean - y) ** 2 for y in ys) / len(ys)

    Fscore = F_mse * (1.0 + LAMBDA * nodes)
    B = B_mse * (1.0 + LAMBDA * 1)
    # Cap below 1000 (i.e. Ratio < 0.92) on EVERY case, not just on average, so no
    # reference solution can saturate a favourable noise draw to a perfect score.
    sc = min(920.0, 100.0 * B / max(1e-9, Fscore))
    print("heldout_MSE=%.6f baseline_MSE=%.6f nodes=%d  Ratio: %.6f"
          % (F_mse, B_mse, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
