#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the volcanic-unrest eruption-probability recovery task.
The solver submits ONE closed-form expression for P(erupt) as a function of the
three logged summary statistics ACC (deformation acceleration), INFL (chamber
inflation index) and SEIS (seismic energy-release rate).

- Reads the case id from <in> (header), then regenerates the hidden law and the
  HELD-OUT episodes -- a markedly MORE INTENSE unrest bout than the training
  catalogue, same underlying magma-volume law -- entirely from that id. The
  law, its coefficients and the held-out episodes are NEVER seen by the
  solver; they live ONLY here.
- Parses the submitted expression with a strict AST whitelist:
      names     ACC INFL SEIS
      operators + - * / **  and unary +/-
      functions sqrt log exp sig tanh absv
      numeric constants
- Evaluates it at every held-out episode's (ACC,INFL,SEIS), clips the result to
  [0,1] (a probability), and scores the mean squared error ("Brier score")
  against the REALIZED (Bernoulli-sampled) held-out outcome -- not the noise-
  free probability -- so even a perfectly recovered law leaves an irreducible
  floor (headroom above the reference solutions).

      metric = mean_i (clip01(p_i) - y_i)^2                      (Brier score)
      O = metric * (1 + LAMBDA * nodes)
      B = baseline_metric * (1 + LAMBDA * 1)   # baseline = constant mean(train y)
      Ratio = min(1000, 100 * B / O) / 1000

A constant predictor reproduces the baseline (~0.1). ACC and SEIS are driven
by the same latent unrest magnitude as INFL, so they look informative on the
training catalogue -- but only INFL (the actual magma volume) determines
whether an episode erupts or fails, and failed intrusions heavily outnumber
eruptions in training. A law that pattern-matches "what accelerating episodes
usually looked like" without weighing how often failed intrusions ALSO looked
that way over-forecasts systematically, and that miscalibration is exposed
hardest on the more intense held-out bout. Non-finite predictions, or any
predictions the grammar can't evaluate, score 0.
"""
import sys, math, ast, random

# ---- fixed design constants (mirrored byte-for-byte in gen.py) ----
SEED_LAW = 700000
SEED_TRAIN = 710000
SEED_HELD = 720000
UTR_LO, UTR_HI = 0.15, 0.75
UHE_LO, UHE_HI = 0.65, 1.35
SIG_A, SIG_S, SIG_I = 0.35, 0.40, 0.35

# ---- held-out / scoring constants (grader only) ----
N_HELD = 250
LAMBDA = 0.003
MAX_NODES = 60
MAX_OUT_BYTES = 100000

ALLOWED_FUNCS = {
    "sqrt": lambda x: math.sqrt(x),
    "log":  lambda x: math.log(x),
    "exp":  lambda x: math.exp(max(-700.0, min(700.0, x))),
    "sig":  lambda x: 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x)))),
    "tanh": math.tanh,
    "absv": abs,
}
ALLOWED_NAMES = {"ACC", "INFL", "SEIS"}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


def n_train(t):
    return 80 + 30 * (t - 1)


# ---------- hidden law (identical to gen.py) ----------
def law(t):
    rng = random.Random(SEED_LAW + t * 91013)
    A0 = rng.uniform(0.6, 1.4); A1 = rng.uniform(3.0, 5.0)
    S0 = rng.uniform(0.4, 1.2); S1 = rng.uniform(2.2, 4.2)
    I0 = rng.uniform(0.2, 0.5); I1 = rng.uniform(1.6, 2.6)
    Vstar = rng.uniform(1.6, 2.2); k1 = rng.uniform(1.8, 3.0)
    return dict(A0=A0, A1=A1, S0=S0, S1=S1, I0=I0, I1=I1, Vstar=Vstar, k1=k1)


def sigmoid(x):
    x = max(-60.0, min(60.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def gen_episodes(t, lo, hi, n, seed, seed_mult):
    p = law(t)
    rng = random.Random(seed + t * seed_mult)
    rows = []
    for _ in range(n):
        U = rng.uniform(lo, hi)
        ACC = p['A0'] + p['A1'] * U + rng.gauss(0.0, SIG_A)
        SEIS = p['S0'] + p['S1'] * U + rng.gauss(0.0, SIG_S)
        INFL = p['I0'] + p['I1'] * U + rng.gauss(0.0, SIG_I)
        pt = sigmoid(p['k1'] * (INFL - p['Vstar']))
        erupt = 1 if rng.random() < pt else 0
        rows.append((ACC, INFL, SEIS, erupt))
    return rows


def gen_train(t):
    return gen_episodes(t, UTR_LO, UTR_HI, n_train(t), SEED_TRAIN, 13)


def gen_held(t):
    """Held-out episodes from a markedly MORE INTENSE unrest bout (higher U range),
    same underlying magma-volume law -- genuine extrapolation, never seen in training."""
    return gen_episodes(t, UHE_LO, UHE_HI, N_HELD, SEED_HELD, 7)


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
    text = lines[-1]                       # take the last non-empty line
    if text.upper().startswith("EXPR "):   # optional leading tag
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


def eval_at(code, ACC, INFL, SEIS):
    env = dict(ALLOWED_FUNCS)
    env["ACC"] = ACC; env["INFL"] = INFL; env["SEIS"] = SEIS
    try:
        p = eval(code, {"__builtins__": {}}, env)
    except Exception:
        return None
    if isinstance(p, bool) or not isinstance(p, (int, float)):
        return None                        # rejects complex results from neg**frac
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
    for ACC, INFL, SEIS, y in held:
        p = eval_at(code, ACC, INFL, SEIS)
        if p is None:
            fail("non-finite / invalid prediction")
        p = max(0.0, min(1.0, p))          # a probability -- clip, don't punish out-of-range
        ds.append((p - y) ** 2)
    metric = sum(ds) / len(ds)

    # baseline: constant predictor = mean of TRAIN outcome (the empirical eruption rate)
    train = gen_train(t)
    mn = sum(r[3] for r in train) / len(train)
    bd = [(mn - y) ** 2 for _, _, _, y in held]
    Bmetric = sum(bd) / len(bd)

    B = Bmetric * (1.0 + LAMBDA * 1)
    O = metric * (1.0 + LAMBDA * nodes)
    sc = min(1000.0, 100.0 * B / max(1e-12, O))
    print("metric=%.6f baseline=%.6f nodes=%d  Ratio: %.6f"
          % (metric, Bmetric, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
