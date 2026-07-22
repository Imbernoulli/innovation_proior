#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for latent tool-wear recursion identification.

- Reads ONLY the test id `t` from <in>'s header (the rest of <in> -- the
  training rows -- is ignored by the grader; the hidden recursion is
  regenerated here, byte-identically to gen.py, from `t` alone).
- The participant submits ONE arithmetic expression string over a fixed
  variable set (Wprev, gap, load, m0, m1, m2, idx) that is meant to BE the
  hidden wear-update recursion `W_i = f(W_{i-1}, gap_i, load_i, material_i)`.
- The grader regenerates several HELD-OUT job sequences -- five times longer
  than training, with a different load / material / idle-gap mix -- under
  the TRUE hidden recursion, rolls the participant's expression forward on
  the SAME sequences (fresh `Wprev = 0` at the start of each), converts both
  the true and the predicted wear trajectories into processing times via the
  GIVEN observation formula, and scores the held-out mean squared error (with
  a small operator-count surcharge) against a "no-wear" baseline:
      F = heldout_MSE(expr)      * (1 + LAMBDA * ops)
      B = heldout_MSE(W == 0 always)
      Ratio = min(1000, 100*B/F) / 1000
  Submitting an expression that stays at exactly zero reproduces B (~0.1).
  Noise and the surcharge keep Ratio = 1.0 unreachable.
"""
import sys, math, random, ast

SEED0 = 926000
SALT_HP = 0
SALT_TRAIN_SEQ = 1
SALT_TRAIN_NOISE = 2
SALT_TEST_SEQ_BASE = 10
SALT_TEST_NOISE_BASE = 50

SIGMA_FRAC = 0.28  # noise sigma is this fraction of the instance's own
                    # (ALPHA * mean BASE) signal scale -- see gen.py.
HELDOUT_SEQS = 3
MAX_T = 10
LAMBDA_OPS = 0.004
MAX_OUT_BYTES = 8000
MAX_EXPR_CHARS = 400
MAX_OPS = 60

ALLOWED_VARS = {"Wprev", "gap", "load", "m0", "m1", "m2", "idx"}
UNARY_FUNCS = {"exp", "log", "sqrt", "abs", "tanh"}
BINARY_FUNCS = {"min", "max"}
ALLOWED_FUNCS = UNARY_FUNCS | BINARY_FUNCS
FUNC_NS = {"exp": math.exp, "log": math.log, "sqrt": math.sqrt,
           "abs": abs, "tanh": math.tanh, "min": min, "max": max}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden recursion (IDENTICAL construction to gen.py) ----------
def rng_for(t, salt):
    return random.Random(SEED0 + t * 104729 + salt * 999983)


def hidden_params(t):
    rng = rng_for(t, SALT_HP)
    decay = rng.uniform(0.08, 0.22)
    wear_rate = rng.uniform(0.006, 0.014)
    mat_mult = [rng.uniform(0.7, 1.0), rng.uniform(1.0, 1.3), rng.uniform(1.3, 1.7)]
    alpha = rng.uniform(0.8, 1.4)
    base = [rng.uniform(8.0, 12.0) for _ in range(3)]
    wcap = 1.0
    return dict(decay=decay, wear_rate=wear_rate, mat_mult=mat_mult,
                alpha=alpha, base=base, wcap=wcap)


def n_train_for(t):
    return 20 + 6 * (t - 1)


def n_test_for(t):
    return 5 * n_train_for(t)


def test_gap(rng):
    if rng.random() < 0.12:
        return rng.uniform(15.0, 30.0)
    return rng.uniform(0.0, 4.0)


def test_load(rng):
    return rng.uniform(5.0, 11.0)


def test_mat(rng):
    return rng.choices([0, 1, 2], weights=[0.2, 0.3, 0.5])[0]


def gen_rows(rng, n, gap_fn, load_fn, mat_fn):
    rows = []
    for i in range(n):
        gap = 0.0 if i == 0 else gap_fn(rng)
        load = load_fn(rng)
        mat = mat_fn(rng)
        rows.append((gap, load, mat))
    return rows


def true_wear_step(W, gap, load, mat, hp):
    Wd = W * math.exp(-hp['decay'] * gap)
    gain = hp['wear_rate'] * hp['mat_mult'][mat] * (load ** 1.5) * (1.0 - Wd / hp['wcap'])
    Wn = Wd + gain
    if Wn < 0.0:
        Wn = 0.0
    if Wn > hp['wcap'] * 2.0:
        Wn = hp['wcap'] * 2.0
    return Wn


def simulate_T(rows, hp, noise_rng, noise_sigma):
    W = 0.0
    Ts = []
    for (gap, load, mat) in rows:
        W = true_wear_step(W, gap, load, mat, hp)
        T = hp['base'][mat] * (1.0 + hp['alpha'] * W) + noise_rng.gauss(0.0, noise_sigma)
        Ts.append(T)
    return Ts


def noise_sigma_for(hp):
    return SIGMA_FRAC * hp['alpha'] * (sum(hp['base']) / 3.0)


def heldout_targets(t, hp):
    n = n_test_for(t)
    sigma = noise_sigma_for(hp)
    targets = []
    for k in range(HELDOUT_SEQS):
        seq_rng = rng_for(t, SALT_TEST_SEQ_BASE + k)
        rows = gen_rows(seq_rng, n, test_gap, test_load, test_mat)
        noise_rng = rng_for(t, SALT_TEST_NOISE_BASE + k)
        Ts = simulate_T(rows, hp, noise_rng, sigma)
        targets.append((rows, Ts))
    return targets


# ---------- participant expression: parse / validate / compile ----------
def validate_and_compile(expr_text):
    if not expr_text or len(expr_text) > MAX_EXPR_CHARS:
        raise ValueError("bad expression length")
    try:
        tree = ast.parse(expr_text, mode="eval")
    except Exception:
        raise ValueError("syntax error")

    op_count = [0]

    def check(node):
        if isinstance(node, ast.Expression):
            check(node.body)
        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
                raise ValueError("disallowed binary operator")
            op_count[0] += 1
            check(node.left)
            check(node.right)
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.UAdd, ast.USub)):
                raise ValueError("disallowed unary operator")
            op_count[0] += 1
            check(node.operand)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCS:
                raise ValueError("disallowed function call")
            if node.keywords:
                raise ValueError("keyword arguments not allowed")
            fname = node.func.id
            need = 2 if fname in BINARY_FUNCS else 1
            if len(node.args) != need:
                raise ValueError("wrong arity for %s" % fname)
            op_count[0] += 1
            for a in node.args:
                check(a)
        elif isinstance(node, ast.Name):
            if node.id not in ALLOWED_VARS:
                raise ValueError("unknown identifier '%s'" % node.id)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("disallowed constant")
        else:
            raise ValueError("disallowed syntax: %s" % type(node).__name__)

    check(tree)
    if op_count[0] > MAX_OPS:
        raise ValueError("expression too complex (%d ops > %d)" % (op_count[0], MAX_OPS))
    code = compile(tree, "<expr>", "eval")
    return code, op_count[0]


def eval_expr(code, variables):
    try:
        val = eval(code, {"__builtins__": {}, **FUNC_NS}, variables)
    except Exception:
        return None
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    val = float(val)
    if not math.isfinite(val):
        return None
    return val


def rollout_predicted_T(code, rows, hp):
    W = 0.0
    preds = []
    for i, (gap, load, mat) in enumerate(rows):
        idx = float(i + 1)
        variables = {
            "Wprev": W, "gap": float(gap), "load": float(load),
            "m0": 1.0 if mat == 0 else 0.0,
            "m1": 1.0 if mat == 1 else 0.0,
            "m2": 1.0 if mat == 2 else 0.0,
            "idx": idx,
        }
        val = eval_expr(code, variables)
        if val is None:
            return None
        W = val
        preds.append(hp['base'][mat] * (1.0 + hp['alpha'] * W))
    return preds


def parse_output(raw):
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() != ""]
    if len(lines) != 1:
        fail("expected exactly one non-empty line")
    return lines[0]


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            header = fh.readline().split()
        t = int(header[0])
    except Exception:
        fail("bad instance header")
    if t < 1 or t > MAX_T:
        fail("bad test id")

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")

    expr_text = parse_output(raw)
    try:
        code, ops = validate_and_compile(expr_text)
    except ValueError as e:
        fail(str(e))

    hp = hidden_params(t)
    targets = heldout_targets(t, hp)

    F_num = 0.0
    B_num = 0.0
    F_den = 0
    for rows, Ts in targets:
        preds = rollout_predicted_T(code, rows, hp)
        if preds is None:
            fail("non-finite value produced during held-out rollout")
        for (gap, load, mat), Ttrue, Tpred in zip(rows, Ts, preds):
            F_num += (Tpred - Ttrue) ** 2
            Tbase = hp['base'][mat]
            B_num += (Tbase - Ttrue) ** 2
            F_den += 1

    F_mse = F_num / F_den
    B_mse = B_num / F_den
    F = F_mse * (1.0 + LAMBDA_OPS * ops)
    B = B_mse
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("heldout_MSE=%.6f baseline_MSE=%.6f ops=%d  Ratio: %.6f"
          % (F_mse, B_mse, ops, sc / 1000.0))


if __name__ == "__main__":
    main()
