#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the buried-crystal dispersion-law recovery task.

- Reads the test id from <in>'s first line, then regenerates the hidden chain
  (omega0=sqrt(K/m), damping gamma, the training band, and the HELD-OUT chain
  sizes/mode indices) entirely from that id -- identical construction to
  gen.py. The hidden pole law lives ONLY here.
- Parses the participant's OUTPUT: a single closed-form Python expression over
  the mode index `j` and the chain size `N`, using +,-,*,/,**,%,//, unary
  +/-, the functions sin/cos/tan/sqrt/exp/log/abs, and the constants pi, e.
  Any other name/call, non-finite constant, empty/oversized text, or a
  non-finite evaluation anywhere on the held-out grid -> Ratio 0.0.
- The expression is evaluated at held-out (j, N) pairs: N far larger than any
  training chain, j spanning low to near-the-zone-edge mode indices -- both a
  size extrapolation and (for most of these pairs) a frequency-band
  extrapolation, since the training sweep only ever covered a partial band.
- Score (minimisation):
      F = heldout_MSE * (1 + LAMBDA * nodes)
      B = baseline_MSE * (1 + LAMBDA * 1)     # baseline = constant omega0
      Ratio = min(1000, 100*B/F) / 1000
  A constant predictor (ignoring j, N) reproduces the baseline (~0.1). A
  higher-harmonic anharmonic correction that ramps up with chain size (zero
  at every training size, so invisible in what the solver is shown) keeps
  even the correct sine pole-law shape below the ceiling on held-out large
  chains, leaving headroom above `strong`.
"""
import sys, math, ast, random

LAMBDA = 0.01
MAX_OUT_BYTES = 4000
MAX_EXPR_CHARS = 300
MAX_NODES = 60

ALLOWED_FUNCS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "sqrt": math.sqrt, "exp": math.exp, "log": math.log, "abs": abs,
}
ALLOWED_CONSTS = {"pi": math.pi, "e": math.e}
J_FRACS = (0.15, 0.35, 0.55, 0.75, 0.95)


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


RAMP_LO, RAMP_HI = 12.0, 50.0
CORR_STRENGTH = 0.15


# ---------- hidden instance (IDENTICAL to gen.py) ----------
def build_instance(test_id):
    rng = random.Random(7000003 * test_id + 911)
    omega0 = rng.uniform(0.8, 2.2)
    gamma = rng.uniform(0.04, 0.10) * omega0
    if test_id <= 3:
        frac_band = rng.uniform(0.62, 0.78)
        n_sizes = rng.randint(3, 4)
        pool = list(range(3, 9))
    elif test_id <= 7:
        frac_band = rng.uniform(0.42, 0.58)
        n_sizes = rng.randint(3, 5)
        pool = list(range(3, 11))
    else:
        frac_band = rng.uniform(0.28, 0.38)
        n_sizes = rng.randint(4, 5)
        pool = list(range(4, 13))
    rng.shuffle(pool)
    N_train = sorted(pool[:n_sizes])
    noise_sigma = rng.uniform(0.03, 0.07)
    ho_pool = [n for n in range(25, 161) if n not in N_train]
    rng.shuffle(ho_pool)
    N_test = sorted(ho_pool[:4])
    return dict(omega0=omega0, gamma=gamma,
                frac_band=frac_band, N_train=N_train,
                noise_sigma=noise_sigma, N_test=N_test)


def true_omega(j, N, omega0):
    x = j * math.pi / (2.0 * (N + 1))
    base = 2.0 * omega0 * math.sin(x)
    ramp = max(0.0, min(1.0, (N - RAMP_LO) / (RAMP_HI - RAMP_LO)))
    corr = CORR_STRENGTH * ramp * 2.0 * omega0 * math.sin(3.0 * x)
    return base + corr


# ---------- safe expression parsing ----------
def _validate(node):
    if isinstance(node, ast.Expression):
        _validate(node.body); return
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div,
                                     ast.Pow, ast.Mod, ast.FloorDiv)):
            raise ValueError("bad binop")
        _validate(node.left); _validate(node.right); return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise ValueError("bad unaryop")
        _validate(node.operand); return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCS:
            raise ValueError("bad call")
        if node.keywords or len(node.args) != 1:
            raise ValueError("bad call arity")
        _validate(node.args[0]); return
    if isinstance(node, ast.Name):
        if node.id in ("j", "N") or node.id in ALLOWED_CONSTS:
            return
        raise ValueError("unknown name %s" % node.id)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("bad constant")
        v = float(node.value)
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("non-finite constant")
        return
    raise ValueError("disallowed syntax %s" % type(node).__name__)


def parse_expr(text):
    text = text.strip()
    if not text:
        raise ValueError("empty expression")
    if len(text) > MAX_EXPR_CHARS:
        raise ValueError("expression too long")
    # single expression only: take the first non-empty line
    text = [ln for ln in text.splitlines() if ln.strip()][0].strip()
    tree = ast.parse(text, mode="eval")
    _validate(tree)
    nodes = sum(1 for nd in ast.walk(tree)
                if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant)))
    if nodes > MAX_NODES:
        raise ValueError("expression too large (%d nodes)" % nodes)
    code = compile(tree, "<expr>", "eval")
    return code, nodes


def eval_expr(code, j, N):
    g = {"__builtins__": {}}
    g.update(ALLOWED_FUNCS)
    g.update(ALLOWED_CONSTS)
    g["j"] = float(j)
    g["N"] = float(N)
    val = eval(code, g, {})
    v = float(val)
    if v != v or v in (float("inf"), float("-inf")):
        raise ValueError("non-finite evaluation")
    return v


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            first = fh.readline().strip()
        test_id = int(first)
    except Exception:
        fail("bad instance header")
    if test_id < 1 or test_id > 100000:
        fail("bad test id")

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace")

    try:
        code, nodes = parse_expr(text)
    except Exception as e:
        fail("invalid expression: %s" % e)

    inst = build_instance(test_id)
    omega0 = inst["omega0"]
    N_test = inst["N_test"]

    ho_points = []
    for Nt in N_test:
        js = sorted(set(max(1, min(Nt, round(f * Nt))) for f in J_FRACS))
        for j in js:
            ho_points.append((j, Nt))

    sq_err = 0.0
    sq_err_b = 0.0
    try:
        for (j, Nt) in ho_points:
            true = true_omega(j, Nt, omega0)
            pred = eval_expr(code, j, Nt)
            sq_err += (pred - true) ** 2
            sq_err_b += (omega0 - true) ** 2
    except Exception as e:
        fail("evaluation error: %s" % e)

    m = len(ho_points)
    F_mse = sq_err / m
    B_mse = sq_err_b / m
    F = F_mse * (1.0 + LAMBDA * nodes)
    B = B_mse * (1.0 + LAMBDA * 1)
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("heldout_MSE=%.6f baseline_MSE=%.6f nodes=%d  Ratio: %.6f"
          % (F_mse, B_mse, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
