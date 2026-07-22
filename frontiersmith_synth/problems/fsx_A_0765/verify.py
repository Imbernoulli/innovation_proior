#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the "ornament beauty law" recovery task. The solver
submits ONE closed-form expression for the beauty score B as a function of
the seven measured columns n, g, D, M, A, K, H.

- Reads the case id from <in> (header), then regenerates the hidden law and
  the HELD-OUT EXTRAPOLATION ledger (bigger grids, finer/varying fold order
  g) entirely from that id. The law and its weights live ONLY here.
- Parses the submitted expression with a strict AST whitelist:
      names     n g D M A K H
      operators + - * / **  and unary +/-
      functions sqrt log exp abs
      numeric constants
- Evaluates it on the held-out ledger, computes a bounded symmetric relative
  error per row, averages, and adds a small node-count parsimony penalty
  (minimise):
      metric = mean_i min(1, |p_i - t_i| / (|p_i| + |t_i| + eps))
      O = metric * (1 + LAMBDA * nodes)
      B = baseline_metric * (1 + LAMBDA * 1)   # baseline = constant mean(train B)
      Ratio = min(1000, 100 * B / O) / 1000
  A constant predictor reproduces the baseline (~0.1). Training only ever
  shows fold order g = 8, so M is a fixed multiple of A throughout training
  and a naive raw regression on D, H, K (ignoring M/A/g entirely) cannot tell
  that the defect term needs to be divided by the ORBIT count M -- not left
  raw and not divided by the motif count K either. The held-out ledger uses
  bigger grids AND finer fold orders (g up to 40), where M pulls sharply away
  from both "no normalisation" and "per-motif" -- only D/M (with the sqrt)
  survives the regime shift. Held-out noise keeps even a correct law below
  the ceiling, leaving headroom above the reference solutions.
"""
import sys, math, ast, random

# ---- fixed design constants (mirrored byte-for-byte in gen.py) ----
N_TRAIN = 70
N_TRAIN_LO, N_TRAIN_HI = 5, 14
G_TRAIN = 8
NOISE_TRAIN = 0.15

# ---- held-out / scoring constants (grader only) ----
N_HELD = 90
N_HELD_LO, N_HELD_HI = 18, 30
G_HELD = [12, 16, 20]
NOISE_HELD = 0.35
LAMBDA = 0.004
CAP = 1.0
MAX_NODES = 60
MAX_OUT_BYTES = 100000

ALLOWED_FUNCS = {
    "sqrt": lambda x: math.sqrt(x),
    "log":  lambda x: math.log(x),
    "exp":  lambda x: math.exp(max(-700.0, min(700.0, x))),
    "abs":  abs,
}
ALLOWED_NAMES = {"n", "g", "D", "M", "A", "K", "H"}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden beauty law (identical to gen.py) ----------
def params(t):
    rng = random.Random(500000 + t * 8161063)
    w1 = rng.uniform(0.8, 1.6)
    w2 = rng.uniform(1.0, 2.4)
    w3 = rng.uniform(1.0, 2.5)
    w4 = rng.uniform(0.6, 1.2)
    return w1, w2, w3, w4


def true_B(D, M, H, K, A, w1, w2, w3, w4):
    return w1 * math.sqrt(D / M) + w2 * H + w3 * (K / A) + w4


def gen_rows(t, n_rows, n_lo, n_hi, g_choices, noise_sigma, seed_base):
    w1, w2, w3, w4 = params(t)
    rng = random.Random(seed_base + t * 131)
    rows = []
    for _ in range(n_rows):
        n = rng.randint(n_lo, n_hi)
        g = rng.choice(g_choices)
        A = n * n
        if g > A:
            g = 1
        M = max(1, A // g)
        rho_D = rng.uniform(0.6, 2.4)
        D = max(1, round(M * rho_D))
        rho_K = rng.uniform(0.03, 0.35)
        K = max(1, round(A * rho_K))
        H = rng.uniform(0.4, 3.2)
        B = true_B(D, M, H, K, A, w1, w2, w3, w4) + rng.gauss(0.0, noise_sigma)
        rows.append((n, g, D, M, A, K, H, B))
    return rows


def gen_train(t):
    return gen_rows(t, N_TRAIN, N_TRAIN_LO, N_TRAIN_HI, [G_TRAIN], NOISE_TRAIN, 111)


def gen_held(t):
    return gen_rows(t, N_HELD, N_HELD_LO, N_HELD_HI, G_HELD, NOISE_HELD, 909001)


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


def eval_at(code, n, g, D, M, A, K, H):
    env = dict(ALLOWED_FUNCS)
    env["n"] = n; env["g"] = g; env["D"] = D; env["M"] = M
    env["A"] = A; env["K"] = K; env["H"] = H
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
    for n, g, D, M, A, K, H, B in held:
        p = eval_at(code, n, g, D, M, A, K, H)
        if p is None:
            fail("non-finite / invalid prediction")
        d = abs(p - B) / (abs(p) + abs(B) + 1e-9)
        ds.append(min(CAP, d))
    metric = sum(ds) / len(ds)

    train = gen_train(t)
    cmean = sum(r[7] for r in train) / len(train)
    bd = [min(CAP, abs(cmean - B) / (abs(cmean) + abs(B) + 1e-9)) for r in held for B in [r[7]]]
    Bmetric = sum(bd) / len(bd)

    Bsc = Bmetric * (1.0 + LAMBDA * 1)
    Osc = metric * (1.0 + LAMBDA * nodes)
    sc = min(1000.0, 100.0 * Bsc / max(1e-12, Osc))
    print("metric=%.6f baseline=%.6f nodes=%d  Ratio: %.6f"
          % (metric, Bmetric, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
