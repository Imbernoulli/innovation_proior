#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the "moons of a far planet keep running late"
eclipse-timing recovery task.

- Reads the test id and the observed TRAIN cycles (k, t_k) from <in>.
- Regenerates the hidden law (t0, P, c, A, M, phi, sigma) and the HELD-OUT
  grading epochs (several years beyond the campaign) entirely from the test
  id -- the hidden law lives ONLY here (and in gen.py's identical copy).
- Parses the participant's closed-form expression for t(k) written over the
  variable `k`, the constant `pi`, arithmetic (+ - * / ** unary -), and the
  unary functions sin, cos, sqrt, exp, log, absv.
- Scores by held-out RMS timing error with a small expression-size
  (parsimony) penalty, normalised against the checker's own trivial
  constant-period (linear ephemeris) fit to the SAME training data:
      F = heldout_RMSE * (1 + LAMBDA * nodes)
      B = baseline_RMSE * (1 + LAMBDA * 5)      # baseline = a + b*k
      Ratio = min(1000, 100*B/F) / 1000
  A linear ephemeris reproduces B (~0.1).  Modelling the bounded wobble only
  helps when it is genuinely resolvable; modelling the *growing* drift is
  what pays off on the far-future grading epochs.
"""
import sys, math, ast, random

BASE = 90210
TRAIN_DAYS = 1400.0
HELD_LO_MULT = 1.7
HELD_HI_MULT = 2.3
P_LO, P_HI = 8.0, 16.0
NH = 26
LAMBDA = 0.003
MAX_NODES = 60
MAX_OUT_BYTES = 20000

ALLOWED_NAMES = {"k", "pi"}
ALLOWED_FUNCS = {
    "sin": math.sin, "cos": math.cos, "sqrt": math.sqrt,
    "exp": lambda x: math.exp(max(-700.0, min(700.0, x))),
    "log": math.log, "absv": abs,
}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden law (identical to gen.py) ----------
def params(t):
    rng = random.Random(BASE + t * 1000003)
    P = rng.uniform(P_LO, P_HI)
    t0 = rng.uniform(0.0, 300.0)
    Ktrainmax = int(TRAIN_DAYS // P)
    trap = (t % 2 == 1)
    if trap:
        M = rng.uniform(1.6, 2.5) * Ktrainmax
        c = rng.uniform(2.2e-4, 4.0e-4)
    else:
        M = rng.uniform(0.12, 0.28) * Ktrainmax
        c = rng.uniform(0.3e-6, 2.0e-6)
    A = rng.uniform(0.6, 1.5)
    phi = rng.uniform(0.0, 2 * math.pi)
    sigma = rng.uniform(0.07, 0.11)
    return dict(P=P, t0=t0, Ktrainmax=Ktrainmax, M=M, c=c, A=A, phi=phi,
                sigma=sigma, trap=trap)


def true_val(k, pr):
    return (pr['t0'] + pr['P'] * k + pr['c'] * k * k
            + pr['A'] * math.sin(2 * math.pi * k / pr['M'] + pr['phi']))


def gen_heldout(t, pr):
    Ktrainmax = pr['Ktrainmax']
    k_lo = max(1, round(HELD_LO_MULT * Ktrainmax))
    k_hi = round(HELD_HI_MULT * Ktrainmax)
    if k_hi <= k_lo:
        k_hi = k_lo + 20
    span = k_hi - k_lo
    n = min(NH, span + 1)
    ks = sorted(set(k_lo + round(i * span / (NH - 1)) for i in range(n)))
    rng4 = random.Random(BASE + 19 + t * 15485867)
    out = [(k, true_val(k, pr) + rng4.gauss(0.0, pr['sigma'])) for k in ks]
    return out


# ---------- expression parsing / validation ----------
_ALLOWED_AST = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
    ast.USub, ast.UAdd,
)


def _validate(tree):
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST):
            return "disallowed syntax %s" % type(node).__name__
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS):
                return "disallowed call"
            if node.keywords or len(node.args) != 1:
                return "bad function arity"
        if isinstance(node, ast.Name):
            if node.id not in ALLOWED_NAMES and node.id not in ALLOWED_FUNCS:
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


def parse_expr(text):
    text = text.strip().splitlines()
    text = text[0].strip() if text else ""
    if not text:
        fail("empty output")
    if len(text) > MAX_OUT_BYTES:
        fail("expression too large")
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


def eval_expr(code, k):
    env = dict(ALLOWED_FUNCS)
    env["k"] = float(k)
    env["pi"] = math.pi
    try:
        v = eval(code, {"__builtins__": {}}, env)
    except Exception:
        fail("evaluation error at k=%d" % k)
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        fail("non-numeric result at k=%d" % k)
    v = float(v)
    if v != v or v in (float("inf"), float("-inf")):
        fail("non-finite result at k=%d" % k)
    return v


# ---------- baseline (trivial linear ephemeris fit to the SAME training data) ----------
def linfit(obs):
    ks = [k for k, _ in obs]
    ts = [v for _, v in obs]
    n = len(ks)
    sk = sum(ks); skk = sum(k * k for k in ks); st = sum(ts)
    skt = sum(k * v for k, v in obs)
    den = n * skk - sk * sk
    if abs(den) < 1e-9:
        b = 0.0
        a = st / n
    else:
        b = (n * skt - sk * st) / den
        a = (st - b * sk) / n
    return a, b


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            lines = fh.read().split("\n")
        header = lines[0].split()
        n_train, t = int(header[0]), int(header[1])
        obs = []
        for ln in lines[1:1 + n_train]:
            parts = ln.split()
            if len(parts) != 2:
                continue
            obs.append((int(parts[0]), float(parts[1])))
        if len(obs) != n_train or n_train < 5:
            fail("bad instance")
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

    pr = params(t)
    held = gen_heldout(t, pr)

    se = 0.0
    for k, true_t in held:
        pred = eval_expr(code, k)
        se += (pred - true_t) ** 2
    F_rmse = math.sqrt(se / len(held))

    a, b = linfit(obs)
    se_b = sum((a + b * k - true_t) ** 2 for k, true_t in held)
    B_rmse = math.sqrt(se_b / len(held))

    F_total = F_rmse * (1.0 + LAMBDA * nodes)
    B_total = B_rmse * (1.0 + LAMBDA * 5)
    sc = min(1000.0, 100.0 * B_total / max(1e-9, F_total))
    print("heldout_RMSE=%.6f baseline_RMSE=%.6f nodes=%d  Ratio: %.6f"
          % (F_rmse, B_rmse, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
