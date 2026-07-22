#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the rug-merchants' haggling-ledger task.

- Reads the test id (and MU) from <in>'s header, then regenerates the SAME
  hidden bazaar law (rho, kappa) that gen.py used for that id, PLUS a
  held-out grading ledger that is genuine EXTRAPOLATION: far larger rival
  counts than training ever showed, and appraisals far outside the training
  range (sometimes both at once). None of this lives in gen.py's stdout.
- Parses the participant's single-line arithmetic expression over n, v
  (+ - * / ** , parentheses, numeric constants, and the two-argument
  functions min(a,b)/max(a,b) only -- no other names or calls) via a
  whitelisted AST walk.
- Evaluates it on every held-out (n, v) pair and scores by mean absolute
  error against the true settled price, normalised against the naive
  "quote-the-full-appraisal" baseline price = v (both get the same fixed
  NOISE_FLOOR added before the ratio -- this is disclosed in statement.md):
      F = MAE(submission)  + NOISE_FLOOR
      B = MAE(price = v)   + NOISE_FLOOR      # baseline: no haggling at all
      Ratio = min(1000, 100*B/max(1e-9,F)) / 1000
  Any parse/evaluation/non-finite failure -> Ratio 0.0.
"""
import sys, math, ast, random

MU = 100.0
N_LIST = (2, 3, 4, 5, 6)
MAX_OUT_BYTES = 20000
MAX_NODES = 60
MAX_CONST_ABS = 1e9
# Fixed additive noise floor added to BOTH F and B before the ratio -- caps how
# far a near-perfectly-identified fit can outrun the naive baseline (an easy
# test id could otherwise recover (rho,kappa) almost exactly and saturate the
# 10x-better-caps-at-1.0 convention). Added identically to F and B so the
# trivial submission (which reproduces B exactly) is completely unaffected.
NOISE_FLOOR = 3.0


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden bazaar law (identical to gen.py) ----------
def hidden_params(t):
    rng = random.Random(7000003 + t * 104729)
    rho = rng.uniform(0.3, 3.0)
    kappa = rng.uniform(0.01, 0.10)
    return rho, kappa


def true_price(n, v, rho, kappa):
    shade = v * (n - 1) / ((n - 1) + rho)
    risk = kappa * (v - MU) ** 2 / MU
    return shade + risk


def held_out_ledger(t, rho, kappa):
    """Regenerated ONLY here; genuine extrapolation on n and/or v. 48 points,
    four 12-point groups: {n=25}, {n=60}, {small n, extreme v}, {n=25/60, extreme v}."""
    rng_ab = random.Random(555000 + t * 15485863)
    rng_cd = random.Random(999000 + t * 22801763)
    sigma = 1.5 + 0.22 * t
    pts = []

    for n_fixed in (25, 60):
        for _ in range(12):
            v = rng_ab.uniform(20.0, 200.0)
            y = true_price(n_fixed, v, rho, kappa) + rng_ab.gauss(0.0, sigma)
            pts.append((n_fixed, v, max(0.01, y)))

    for group_n_choices in (N_LIST, (25, 60)):
        for i in range(12):
            n = rng_cd.choice(group_n_choices)
            v = rng_cd.uniform(5.0, 15.0) if i % 2 == 0 else rng_cd.uniform(300.0, 500.0)
            y = true_price(n, v, rho, kappa) + rng_cd.gauss(0.0, sigma)
            pts.append((n, v, max(0.01, y)))

    return pts


# ---------- expression parsing / validation ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load, ast.Call,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
    ast.USub, ast.UAdd,
)
_ALLOWED_NAMES = {"n", "v"}
_ALLOWED_FUNCS = {"min": (lambda a, b: a if a <= b else b),
                   "max": (lambda a, b: a if a >= b else b)}


def _count_nodes(tree):
    return sum(1 for nd in ast.walk(tree)
               if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Name, ast.Constant, ast.Call)))


def parse_expression(raw):
    text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        fail("empty output")
    if len(lines) != 1:
        fail("expected exactly one non-blank line")
    expr_text = lines[0]
    if len(expr_text) > 2000:
        fail("expression too long")
    try:
        tree = ast.parse(expr_text, mode="eval")
    except Exception:
        fail("parse error")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            fail("disallowed syntax %s" % type(node).__name__)
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS):
                fail("disallowed call")
            if node.keywords or len(node.args) != 2:
                fail("min/max need exactly 2 positional args")
        if isinstance(node, ast.Name):
            if node.id in _ALLOWED_FUNCS:
                continue
            if node.id not in _ALLOWED_NAMES:
                fail("unknown name '%s'" % node.id)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                fail("non-numeric constant")
            cv = float(node.value)
            if cv != cv or cv in (float("inf"), float("-inf")):
                fail("non-finite constant")
            if abs(cv) > MAX_CONST_ABS:
                fail("constant magnitude too large")
    nodes = _count_nodes(tree)
    if nodes > MAX_NODES:
        fail("expression too large (%d nodes)" % nodes)
    try:
        code = compile(tree, "<expr>", "eval")
    except Exception:
        fail("compile error")
    return code


def evaluate(code, n, v):
    glob = {"__builtins__": {}}
    env = {"n": float(n), "v": float(v)}
    env.update(_ALLOWED_FUNCS)
    try:
        p = eval(code, glob, env)
    except Exception:
        fail("evaluation error")
    if not isinstance(p, (int, float)) or isinstance(p, bool):
        fail("non-numeric result")
    p = float(p)
    if p != p or p in (float("inf"), float("-inf")):
        fail("non-finite result")
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
    if t < 1 or t > 1000000:
        fail("bad test id")

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")

    code = parse_expression(raw)

    rho, kappa = hidden_params(t)
    ledger = held_out_ledger(t, rho, kappa)

    ae_sum = 0.0
    base_ae_sum = 0.0
    for n, v, y in ledger:
        p = evaluate(code, n, v)
        ae_sum += abs(p - y)
        base_ae_sum += abs(v - y)

    F = ae_sum / len(ledger) + NOISE_FLOOR
    B = base_ae_sum / len(ledger) + NOISE_FLOOR
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("heldout_MAE=%.6f baseline_MAE=%.6f n_heldout=%d  Ratio: %.6f"
          % (F, B, len(ledger), sc / 1000.0))


if __name__ == "__main__":
    main()
