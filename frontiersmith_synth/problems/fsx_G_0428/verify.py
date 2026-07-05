#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the integer-sequence oracle (recurrence discovery).

- Reads the test id from <in> (first line), then regenerates the hidden law,
  the training prefix and the FAR-FUTURE held-out window entirely from that id
  (the law lives ONLY here -- never shipped, never printed by gen.py).
- Parses the participant's closed-form / recurrence expression from <out>
  through a strict AST whitelist.  Variables:
        n           -- the (integer) index of the term being predicted
        a1, a2, a3  -- the true previous terms a(n-1), a(n-2), a(n-3)
                       (teacher forcing: supplied from the far-future ground
                        truth, NOT from the participant's own predictions)
  Allowed functions: exp log sin cos sqrt tanh abs ; operators + - * / ** % .
  Imports / attributes / unknown names / non-finite results are rejected -> 0.
- Score (minimisation, complexity-penalised relative held-out error):
        err(n) = |pred(n) - true(n)| / (1 + |true(n)|)
        F = mean_heldout err * (1 + LAMBDA * complexity)
        B = baseline_err     * (1 + LAMBDA * 1)   # baseline = constant last term
        Ratio = min(1000, 100*B/F) / 1000
  A constant equal to the last observed term reproduces the baseline (~0.1);
  recovering the underlying law drives the error down toward the irreducible
  jitter floor, but that floor (fresh far-future jitter the model cannot see)
  keeps even a strong recovery below 1.0.
"""
import sys, math, ast, random

LAMBDA = 0.002
MAX_EXPR_BYTES = 200000

# ---- ladder configuration (identical to gen.py) ----
TLIST  = [24, 24, 22, 22, 20, 20, 18, 18, 16, 16]
DELTAS = [0.26, 0.20, 0.28, 0.28, 0.22, 0.30, 0.30, 0.24, 0.32, 0.34]
GAP    = 6
NHELD  = 14

LAW_BASE      = 770001
PRIME         = 100003
TRAIN_NOISE   = 880002
HELDOUT_NOISE = 990003

TEMPLATES = [
    ("fib",   2, [1, 1]),
    ("poly2", 3, [3, -3, 1]),
    ("fib",   2, [1, 1]),
    ("trib",  3, [1, 1, 1]),
    ("poly3", 4, [4, -6, 4, -1]),
    ("trib",  3, [1, 1, 1]),
    ("tetra", 4, [1, 1, 1, 1]),
    ("poly3", 4, [4, -6, 4, -1]),
    ("penta", 5, [1, 1, 1, 1, 1]),
    ("penta", 5, [1, 1, 1, 1, 1]),
]

ALLOWED_FUNCS = {"exp": math.exp, "log": math.log, "sin": math.sin,
                 "cos": math.cos, "sqrt": math.sqrt, "tanh": math.tanh,
                 "abs": abs}
ALLOWED_VARS = {"n", "a1", "a2", "a3"}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---- ground truth (identical logic to gen.py) ----
def make_seeds(kind, K, rng):
    if kind == "poly2":
        A = rng.randint(1, 3); B = rng.randint(0, 3); C = rng.randint(1, 4)
        return [A * k * k + B * k + C for k in range(K)]
    if kind == "poly3":
        A = rng.randint(1, 2); B = rng.randint(0, 2)
        C = rng.randint(0, 3); D = rng.randint(1, 4)
        return [A * k ** 3 + B * k * k + C * k + D for k in range(K)]
    return [rng.randint(1, 5) for _ in range(K)]


def clean_signal(t, maxidx):
    kind, K, coeffs = TEMPLATES[(t - 1) % len(TEMPLATES)]
    rng = random.Random(LAW_BASE + t * PRIME)
    seeds = make_seeds(kind, K, rng)
    L = list(seeds)
    for n in range(K, maxidx + 1):
        L.append(sum(coeffs[i] * L[n - 1 - i] for i in range(K)))
    return L


def build_true(t):
    T = TLIST[(t - 1) % len(TLIST)]
    delta = DELTAS[(t - 1) % len(DELTAS)]
    maxidx = T + GAP + NHELD - 1
    L = clean_signal(t, maxidx)
    tr_rng = random.Random(TRAIN_NOISE + t * PRIME)
    ho_rng = random.Random(HELDOUT_NOISE + t * PRIME)
    true = []
    for n in range(maxidx + 1):
        rng = tr_rng if n < T else ho_rng
        j = rng.uniform(-delta, delta)
        true.append(int(round(L[n] * (1.0 + j))))
    held = [T + GAP + i for i in range(NHELD)]
    return T, true, held


# ---- strict expression validation ----
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.USub, ast.UAdd,
)


def validate_ast(tree):
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return "disallowed syntax %s" % type(node).__name__
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS):
                return "disallowed call"
            if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
                return "bad call args"
        if isinstance(node, ast.Name):
            if node.id not in ALLOWED_VARS and node.id not in ALLOWED_FUNCS:
                return "unknown name %s" % node.id
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            return "non-numeric constant"
    return None


def complexity(tree):
    return sum(1 for nd in ast.walk(tree)
               if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant)))


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
            raw = fh.read(MAX_EXPR_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_EXPR_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) != 1:
        fail("expression must be a single non-empty line")
    expr = lines[0].strip()

    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        fail("parse error")
    reason = validate_ast(tree)
    if reason:
        fail(reason)

    # Coerce every numeric literal to float so that '**' can never build a giant
    # arbitrary-precision integer (e.g. 9**9**9**9) and hang the grader: float
    # overflow raises/returns inf instantly and is rejected below.
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, int) \
                and not isinstance(node.value, bool):
            node.value = float(node.value)
    ast.fix_missing_locations(tree)

    try:
        code = compile(tree, "<expr>", "eval")
    except Exception:
        fail("compile error")

    T, true, held = build_true(t)
    cx = complexity(tree)

    # participant evaluation on the far-future window (teacher forcing)
    tot = 0.0
    for n in held:
        env = {"n": float(n),
               "a1": float(true[n - 1]),
               "a2": float(true[n - 2]),
               "a3": float(true[n - 3])}
        env.update(ALLOWED_FUNCS)
        try:
            p = eval(code, {"__builtins__": {}}, env)
        except Exception:
            fail("evaluation error")
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            fail("non-numeric result")
        p = float(p)
        if not math.isfinite(p):
            fail("non-finite result")
        tot += abs(p - true[n]) / (1.0 + abs(true[n]))
    F_err = tot / len(held)

    # internal baseline: predict the last observed term as a constant
    c = true[T - 1]
    B_err = sum(abs(c - true[n]) / (1.0 + abs(true[n])) for n in held) / len(held)

    B = B_err * (1.0 + LAMBDA * 1)
    F = F_err * (1.0 + LAMBDA * cx)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("heldout_err=%.6f baseline_err=%.6f complexity=%d  Ratio: %.6f"
          % (F_err, B_err, cx, sc / 1000.0))


if __name__ == "__main__":
    main()
