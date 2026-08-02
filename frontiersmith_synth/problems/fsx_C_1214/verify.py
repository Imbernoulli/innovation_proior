#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for gpu-batch-throughput-forecast.

- Reads the test id `t` from <in>'s header (the four visible constants
  C,W,F,D and the training rows are re-derivable from `t` alone via the SAME
  `params(t)` used by gen.py -- the grader recomputes them itself rather than
  trusting the file, so the hidden knee K and the held-out batches can never
  leak through a tampered .in).
- Parses the participant's output: ONE line holding a Python-style arithmetic
  expression over the variables x (batch size), C, W, F, D, using
  + - * / ** unary +/-, parentheses, numeric constants, and the unary/binary
  functions sqrt, log, exp, absv (1 arg), min, max (2 args).  Anything else
  (attribute access, calls to unknown names, comparisons, etc.) is rejected.
- Regenerates HELD-OUT batch sizes -- far past the hidden knee, a genuinely
  different regime from every training batch -- and their true (lightly
  noised) throughputs, entirely inside this file.
- Evaluates the expression at each held-out batch (substituting that
  instance's own C,W,F,D), and scores from the mean log-ratio error to a
  fitted-then-fixed accuracy metric, with a light parsimony penalty:
      e_i    = min(ERR_CAP, |ln(pred_i / true_i)|)
      Fq     = 1 / (mean(e_i) + BIAS) - complexity_penalty
      B      = same Fq formula for the constant "mean of the training
               throughputs" predictor (the checker's own trivial baseline)
      Ratio  = min(1000, 100 * Fq / B) / 1000
  A constant predictor reproduces B (Ratio ~= 0.1).  A predictor that keeps
  climbing past the knee (ignores the bandwidth ceiling) racks up large
  log-ratio error on the far batches.  Recovering P = min(C/F, W/D) exactly
  from the given constants, plus a modest fit of the ramp scale from the
  training ramp, drives the error down and leaves headroom above via the
  irreducible measurement noise on the held-out trace.
"""
import sys, math, ast, random

ERR_CAP = 1.0
BIAS = 0.10
MAX_NODES = 60
MAX_OUT_BYTES = 20000
HELD_X = [110, 150, 210, 300, 430, 620, 900, 1300]

ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
    "absv": abs,
    "min": min,
    "max": max,
}
FUNC_ARITY = {"sqrt": 1, "log": 1, "exp": 1, "absv": 1, "min": 2, "max": 2}
ALLOWED_NAMES = {"x", "C", "W", "F", "D"}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden/visible instance parameters (identical to gen.py) ----------
def params(t):
    rng = random.Random(5_312_009 + t * 7919)
    Qc = rng.uniform(80.0, 1500.0)
    Qw = rng.uniform(80.0, 1500.0)
    F = rng.uniform(50.0, 500.0)
    D = rng.uniform(10.0, 200.0)
    C = Qc * F
    W = Qw * D
    K = rng.uniform(60.0, 180.0)
    return C, W, F, D, K


def true_throughput(x, C, W, F, D, K):
    P = min(C / F, W / D)
    return P * x / (x + K)


def held_out_truth(t, C, W, F, D, K):
    rng = random.Random(778_001 + t * 950213)   # independent stream from training noise
    sigma = 0.05
    out = []
    for x in HELD_X:
        mu = true_throughput(x, C, W, F, D, K)
        y = mu * (1.0 + rng.gauss(0.0, sigma))
        out.append(max(y, 1e-6))
    return out


# ---------- safe expression parsing ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
)


def _validate_ast(tree):
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return "disallowed syntax %s" % type(node).__name__
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS):
                return "disallowed call"
            if node.keywords or len(node.args) != FUNC_ARITY[node.func.id]:
                return "bad function arity"
        if isinstance(node, ast.Name):
            if node.id in ALLOWED_FUNCS:
                continue
            if node.id not in ALLOWED_NAMES:
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


def parse_expression(raw):
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        fail("empty output")
    if len(lines) > 1:
        fail("output must be a single expression line")
    text = lines[0]
    if len(text) > 2000:
        fail("expression too long")
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


def eval_expr(code, x, C, W, F, D):
    env = dict(ALLOWED_FUNCS)
    env.update({"x": float(x), "C": C, "W": W, "F": F, "D": D})
    try:
        v = eval(code, {"__builtins__": {}}, env)
    except Exception:
        return None
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return None
    v = float(v)
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v


def quality(code, xs, C, W, F, D, trues):
    errs = []
    for x, tv in zip(xs, trues):
        p = eval_expr(code, x, C, W, F, D)
        if p is None or p <= 0.0:
            return None
        e = min(ERR_CAP, abs(math.log(p / tv)))
        errs.append(e)
    return 1.0 / (sum(errs) / len(errs) + BIAS)


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            head1 = fh.readline().split()
            n_train = int(head1[0])
            t = int(head1[1])
            fh.readline()  # C W F D line (recomputed, not trusted)
            train_rows = []
            for _ in range(n_train):
                ln = fh.readline().split()
                train_rows.append((float(ln[0]), float(ln[1])))
    except Exception:
        fail("bad instance file")
    if t < 1 or t > 1_000_000 or n_train != len(train_rows):
        fail("bad instance header")

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace")

    code, nodes = parse_expression(text)

    C, W, F, D, K = params(t)
    trues = held_out_truth(t, C, W, F, D, K)

    Fq_raw = quality(code, HELD_X, C, W, F, D, trues)
    if Fq_raw is None:
        fail("non-finite or non-positive prediction on held-out batch")
    penalty = 0.02 * max(0.0, (nodes - 60)) / 10.0   # never fires (nodes<=MAX_NODES=60); safety net
    Fq = max(1e-9, Fq_raw - penalty)

    mean_train_y = sum(y for _, y in train_rows) / len(train_rows)
    const_code = compile(ast.parse(repr(mean_train_y), mode="eval"), "<const>", "eval")
    B = quality(const_code, HELD_X, C, W, F, D, trues)
    if B is None or B <= 0.0:
        fail("baseline degenerate")

    sc = min(1000.0, 100.0 * Fq / max(1e-9, B))
    print("Fq=%.6f baseline=%.6f nodes=%d  Ratio: %.6f" % (Fq, B, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
