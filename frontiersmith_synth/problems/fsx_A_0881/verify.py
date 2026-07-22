#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the smith's-quench-log law-recovery task.

- Reads the header from <in> (n_train, test id, ambient) and the training
  rows.  Regenerates the hidden law's coefficients (a, b) from the test id
  ONLY -- the same pure function as gen.py's params(), duplicated here (never
  imported) so no ground-truth module ships in the directory.
- Parses the participant's single-line arithmetic EXPRESSION over the
  variable T (operators + - * / **, numeric constants, parentheses; ** only
  with a small non-negative integer literal exponent).  Any parse failure,
  disallowed syntax, or non-finite evaluation anywhere in scoring -> Ratio 0.
- Scores prediction accuracy on TWO deterministically regenerated point sets:
    * TRAIN-region fresh points (same range as the log, unseen exact values) -- weight 0.3
    * HELD-OUT points, far hotter than anything ever logged (genuine
      extrapolation; regenerated only here)                                  -- weight 0.7
  Per point accuracy = max(0, 1 - |pred-true| / (|true|+eps)); F is the
  weighted mean.  The internal baseline B is the SAME blended metric for the
  constant predictor "mean of the training dT column" (a trivial submission
  reproduces this exactly -> Ratio ~= 0.1).
      Ratio = min(1000, 100*F/max(1e-9,B)) / 1000
  Because the cubic term is a minor, noise-swamped correction on the logged
  range, a model that only explains the TRAIN region well (high in-sample
  goodness-of-fit) can still miss it entirely and collapse on the held-out,
  radiation-dominated regime.
"""
import sys, ast, random, math

AMBIENT = 20.0
TRAIN_LO, TRAIN_HI = 30.0, 320.0
HELD_LO, HELD_HI = 800.0, 1500.0
EPS = 1e-3
W_TRAIN, W_HELD = 0.3, 0.7
N_EVAL_TRAIN, N_EVAL_HELD = 300, 300
MAX_EXPR_CHARS = 200
MAX_OUT_BYTES = 20000
MAX_POW_EXP = 6


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden law (identical to gen.py; not imported) ----------
def params(t):
    rng = random.Random(900191 + t * 7919)
    a = rng.uniform(0.020, 0.050)
    frac_hi = 0.35 - 0.025 * (t - 1)
    frac_lo = max(0.03, frac_hi - 0.09)
    cubic_frac = rng.uniform(frac_lo, frac_hi)
    td = TRAIN_HI - AMBIENT
    b = cubic_frac * a / (td * td)
    noise_mult = rng.uniform(2.0, 3.0)
    sigma = noise_mult * b * (td ** 3)
    n = 2000 + 1200 * (t - 1)
    return a, b, sigma, n


def true_delta(T, a, b):
    x = T - AMBIENT
    return -(a * x + b * x ** 3)


# ---------- safe expression parsing (variable T only) ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
    ast.USub, ast.UAdd,
)


def _validate(tree):
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return "disallowed syntax %s" % type(node).__name__
        if isinstance(node, ast.Name) and node.id != "T":
            return "unknown name %s" % node.id
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                return "non-numeric constant"
            try:
                v = float(node.value)
            except OverflowError:
                return "constant magnitude too large"
            if v != v or v in (float("inf"), float("-inf")):
                return "non-finite constant"
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            exp = node.right
            if not (isinstance(exp, ast.Constant) and isinstance(exp.value, int)
                    and not isinstance(exp.value, bool) and 0 <= exp.value <= MAX_POW_EXP):
                return "** exponent must be an integer literal in [0,%d]" % MAX_POW_EXP
    return None


def compile_expr(text):
    text = text.strip()
    if not text:
        fail("empty expression")
    if len(text) > MAX_EXPR_CHARS:
        fail("expression too long")
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    err = _validate(tree)
    if err:
        fail(err)
    try:
        return compile(tree, "<expr>", "eval")
    except Exception:
        fail("compile error")


def eval_at(code, T):
    try:
        v = eval(code, {"__builtins__": {}}, {"T": T})
    except Exception:
        fail("evaluation error")
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        fail("non-numeric result")
    try:
        v = float(v)
    except OverflowError:
        fail("result magnitude too large")
    if v != v or v in (float("inf"), float("-inf")):
        fail("non-finite result")
    return v


# ---------- scoring ----------
def accuracy(pred_code, points, a, b):
    total = 0.0
    for T in points:
        true = true_delta(T, a, b)
        pred = eval_at(pred_code, T)
        relerr = abs(pred - true) / (abs(true) + EPS)
        total += max(0.0, 1.0 - relerr)
    return total / len(points)


def const_accuracy(const_val, points, a, b):
    total = 0.0
    for T in points:
        true = true_delta(T, a, b)
        relerr = abs(const_val - true) / (abs(true) + EPS)
        total += max(0.0, 1.0 - relerr)
    return total / len(points)


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            lines = fh.read().split("\n")
        header = lines[0].split()
        n = int(header[0]); t = int(header[1]); ambient = float(header[2])
        train_deltas = []
        for i in range(n):
            parts = lines[1 + i].split()
            train_deltas.append(float(parts[1]))
    except Exception:
        fail("bad instance file")
    if t < 1 or t > 100000 or n <= 0 or abs(ambient - AMBIENT) > 1e-6:
        fail("bad instance header")

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace").strip()
    if "\n" in text:
        # only the first non-empty line is the expression; trailing junk not allowed
        nonempty = [ln for ln in text.splitlines() if ln.strip()]
        if len(nonempty) != 1:
            fail("output must be exactly one non-empty line")
        text = nonempty[0]

    code = compile_expr(text)

    a, b, sigma, n_expected = params(t)

    rng_train = random.Random(555 + t * 97)
    train_pts = [rng_train.uniform(TRAIN_LO, TRAIN_HI) for _ in range(N_EVAL_TRAIN)]
    rng_held = random.Random(9999 + t * 97)
    held_pts = [rng_held.uniform(HELD_LO, HELD_HI) for _ in range(N_EVAL_HELD)]

    acc_train = accuracy(code, train_pts, a, b)
    acc_held = accuracy(code, held_pts, a, b)
    F = W_TRAIN * acc_train + W_HELD * acc_held

    const_val = sum(train_deltas) / len(train_deltas)
    b_train = const_accuracy(const_val, train_pts, a, b)
    b_held = const_accuracy(const_val, held_pts, a, b)
    B = W_TRAIN * b_train + W_HELD * b_held

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("train_acc=%.6f held_acc=%.6f baseline=%.6f  Ratio: %.6f"
          % (acc_train, acc_held, B, sc / 1000.0))


if __name__ == "__main__":
    main()
