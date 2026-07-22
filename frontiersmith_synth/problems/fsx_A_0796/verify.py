#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the "lighthouse relay" transducer-induction task.
The solver submits ONE closed-form expression for the register reading y as
a function of the ship count n and the currently-engaged gear s = n mod 3.

- Reads the case id from <in> (header), then regenerates the hidden per-gear
  rational law and the HELD-OUT EXTRAPOLATION log (ship counts 200x-40000x
  larger than anything in training) entirely from that id. The law and its
  weights live ONLY here.
- Parses the submitted expression with a strict AST whitelist:
      names     n s
      operators + - * / **  and unary +/-
      functions sqrt log exp abs
      numeric constants
- Evaluates it on the held-out log, computes a bounded symmetric relative
  error per row, averages, and adds a small node-count parsimony penalty
  (minimise):
      metric = mean_i min(1, |p_i - t_i| / (|p_i| + |t_i| + eps))
      O = metric * (1 + LAMBDA * nodes)
      B = baseline_metric * (1 + LAMBDA * 1)   # baseline = constant mean(train y)
      Ratio = min(1000, 100 * B / O) / 1000
  A constant predictor reproduces the baseline (~0.1). Fitting a SINGLE
  pooled rational curve that ignores which gear is engaged (the natural
  "textbook" move: cross-multiply, one linear system, one curve) captures
  the *shape* (bounded, not exploding) but lands on one compromise asymptote
  -- systematically wrong for whichever gears its curve doesn't happen to
  match, since gear identity and ship count interact and training's modest
  counts never force that interaction into view. A per-gear rational
  recovery generalises because a Mobius map is determined (up to scale) by
  finitely many exact points, independent of how far n is extrapolated.
  Held-out noise keeps even a correct law below the ceiling, leaving
  headroom above the reference solutions.
"""
import sys, math, ast, random

# ---- fixed design constants (mirrored byte-for-byte in gen.py) ----
K = 3
N_TRAIN = 90
N_TRAIN_LO, N_TRAIN_HI = 4, 90
NOISE_TRAIN = 0.04

# ---- held-out / scoring constants (grader only) ----
N_HELD = 60
N_HELD_LO, N_HELD_HI = 20000, 400000
NOISE_HELD = 0.12
LAMBDA = 0.003
CAP = 1.0
MAX_NODES = 150
MAX_OUT_BYTES = 100000

ALLOWED_FUNCS = {
    "sqrt": lambda x: math.sqrt(x),
    "log":  lambda x: math.log(x),
    "exp":  lambda x: math.exp(max(-700.0, min(700.0, x))),
    "abs":  abs,
}
ALLOWED_NAMES = {"n", "s"}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden per-gear rational law (identical to gen.py) ----------
def gear_params(t):
    rng = random.Random(700000 + t * 9176321)
    ps = []
    for _s in range(K):
        a = rng.randint(3, 9)
        b = rng.randint(0, 20)
        c = rng.randint(1, 4)
        d = rng.randint(100, 600)
        ps.append((a, b, c, d))
    return ps


def true_y(n, s, ps):
    a, b, c, d = ps[s]
    return (a * n + b) / (c * n + d)


def gen_rows(t, n_rows, n_lo, n_hi, noise_sigma, seed_base):
    ps = gear_params(t)
    rng = random.Random(seed_base + t * 131)
    rows = []
    for _ in range(n_rows):
        n = rng.randint(n_lo, n_hi)
        s = n % K
        y = true_y(n, s, ps) * (1.0 + rng.gauss(0.0, noise_sigma))
        rows.append((n, s, y))
    return rows


def gen_train(t):
    return gen_rows(t, N_TRAIN, N_TRAIN_LO, N_TRAIN_HI, NOISE_TRAIN, 111)


def gen_held(t):
    return gen_rows(t, N_HELD, N_HELD_LO, N_HELD_HI, NOISE_HELD, 909001)


# ---------- expression parsing / validation ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
    ast.USub, ast.UAdd,
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
        if isinstance(node, ast.Name):
            if node.id in ALLOWED_FUNCS or node.id in ALLOWED_NAMES:
                continue
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


def parse_expr(raw):
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        fail("empty submission")
    text = lines[-1]
    if text.upper().startswith("EXPR "):
        text = text[5:].strip()
    if not text:
        fail("empty expression")
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


def eval_at(code, n, s):
    env = dict(ALLOWED_FUNCS)
    env["n"] = n
    env["s"] = s
    try:
        p = eval(code, {"__builtins__": {}}, env)
    except Exception:
        return None
    if isinstance(p, bool) or not isinstance(p, (int, float)):
        return None
    p = float(p)
    if p != p or p in (float("inf"), float("-inf")):
        return None
    return p


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
    if t < 1 or t > 100000:
        fail("bad test id")

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace")

    code, nodes = parse_expr(text)

    held = gen_held(t)
    ds = []
    for n, s, y in held:
        p = eval_at(code, n, s)
        if p is None:
            fail("non-finite / invalid prediction")
        d = abs(p - y) / (abs(p) + abs(y) + 1e-9)
        ds.append(min(CAP, d))
    metric = sum(ds) / len(ds)

    train = gen_train(t)
    cmean = sum(r[2] for r in train) / len(train)
    bd = [min(CAP, abs(cmean - y) / (abs(cmean) + abs(y) + 1e-9)) for r in held for y in [r[2]]]
    Bmetric = sum(bd) / len(bd)

    Bsc = Bmetric * (1.0 + LAMBDA * 1)
    Osc = metric * (1.0 + LAMBDA * nodes)
    sc = min(1000.0, 100.0 * Bsc / max(1e-12, Osc))
    print("metric=%.6f baseline=%.6f nodes=%d  Ratio: %.6f"
          % (metric, Bmetric, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
