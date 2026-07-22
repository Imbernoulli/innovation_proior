#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the springback-regime-law recovery task.

- Reads the test id from <in> (header), then regenerates the hidden two-regime
  law (elastic slope c1, plastic coefficients c2/c3, and the t-dependent
  boundary A - Bc*t) and the HELD-OUT THICK-SHEET sample entirely from that id
  -- identical formulas to gen.py, but a DIFFERENT thickness range, so the
  dominant regime flips relative to what the solver trained on. The hidden law
  lives ONLY here (and in gen.py); it is never printed to the solver.
- Parses the participant's closed-form expression: arithmetic over the two
  input variables `r` (die radius) and `t` (sheet thickness), the operators
  + - * / ** and parentheses, numeric constants, and the unary functions
  step, sig, relu, absv, sqrt.
- Evaluates the expression row-by-row on the held-out sample and scores by
  held-out MSE with a small node-count parsimony penalty (minimisation):
      F = heldout_MSE * (1 + LAMBDA * nodes)
      B = baseline_MSE * (1 + LAMBDA * 1)      # baseline = constant 1.2
      Ratio = min(1000, 100*B/F) / 1000
  A constant reproduces the baseline (~0.1). The held-out sample is a genuine
  MIX of both regimes (not saturated to one branch), so neither "assume
  elastic everywhere" nor "assume plastic everywhere" -- nor any single
  smooth curve in r/t alone, however shaped -- can fit it: the SAME r/t ratio
  is elastic-dominated for thin sheets and plastic-saturated for thick ones,
  and only branching on the recovered t-dependent boundary gets both parts
  right. Sensor noise keeps even a good model off the ceiling, leaving
  headroom.
"""
import sys, math, ast, random

LAMBDA = 0.004
BASELINE_CONST = 1.2
NH = 420
# Held out on THICKER sheets than train (0.5-1.0), but NOT so thick that the
# boundary xc(t)=A-Bc*t collapses to its floor across the whole range -- that
# would make held-out ~100% one regime and let a boundary-blind global fit of
# just the dominant branch win by luck. (1.4, 2.2) keeps a genuine two-regime
# MIX in held-out (roughly 15-40% elastic across cases/testIds), so only a
# predictor that tracks the t-dependent boundary itself gets both parts right.
T_HOLD = (1.4, 2.2)
X_RANGE = (2.0, 30.0)   # r/t ratio range -- SAME domain as gen.py's training split
XC_FLOOR = 0.5
MAX_NODES = 60
MAX_OUT_BYTES = 20000
BASE = 918273


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden law (identical to gen.py) ----------
def hidden_params(cid):
    rng = random.Random(BASE + cid * 104729)
    A = 30.0 + rng.uniform(-2.0, 2.0)
    Bc = 11.0 + rng.uniform(-1.5, 1.5)
    c1 = rng.uniform(0.055, 0.095)
    c2 = rng.uniform(0.85, 1.25)
    c3 = rng.uniform(-0.12, 0.12)
    return A, Bc, c1, c2, c3


def xc(A, Bc, t):
    return max(XC_FLOOR, A - Bc * t)


def true_S(r, t, A, Bc, c1, c2, c3):
    x = r / t
    if x < xc(A, Bc, t):
        return c1 * x
    return c2 * (x ** (1.0 / 3.0)) + c3


def heldout_sample(cid, A, Bc, c1, c2, c3):
    sigma = 0.40 + 0.050 * (cid - 1)
    rng_s = random.Random(70003 + cid * 15485863)   # DIFFERENT stream from gen.py's train
    rng_n = random.Random(60013 + cid * 87178291)
    rows = []
    for _ in range(NH):
        t = rng_s.uniform(*T_HOLD)
        x = rng_s.uniform(*X_RANGE)
        r = x * t
        s_true = true_S(r, t, A, Bc, c1, c2, c3)
        s_obs = s_true + rng_n.gauss(0.0, sigma)
        rows.append((r, t, s_obs))
    return rows


# ---------- expression DSL ----------
ALLOWED_FUNCS = {
    "step": lambda x: 1.0 if x > 0 else 0.0,
    "sig": lambda x: 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x)))),
    "relu": lambda x: x if x > 0 else 0.0,
    "absv": abs,
    "sqrt": lambda x: math.sqrt(x) if x >= 0 else _domain_err(),
}


class _DomainError(Exception):
    pass


def _domain_err():
    raise _DomainError("domain")


_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
)
_ALLOWED_NAMES = {"r", "t"}


def _validate_ast(tree):
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return "disallowed syntax %s" % type(node).__name__
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS):
                return "disallowed call"
            if node.keywords or len(node.args) != 1:
                return "bad function arity"
        if isinstance(node, ast.Name):
            if node.id in ALLOWED_FUNCS:
                continue
            if node.id not in _ALLOWED_NAMES:
                return "unknown name %s" % node.id
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                return "non-numeric constant"
            v = float(node.value)
            if v != v or v in (float("inf"), float("-inf")):
                return "non-finite constant"
            if abs(v) > 1e6:
                return "constant out of range"
    return None


def _count_nodes(tree):
    return sum(1 for nd in ast.walk(tree)
               if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant)))


def parse_expr(raw):
    text = raw.strip()
    if not text:
        fail("empty output")
    if "\n" in text.strip("\n"):
        # allow a single trailing newline but not multiple statements/lines
        first_nonempty = [ln for ln in text.splitlines() if ln.strip()]
        if len(first_nonempty) != 1:
            fail("expected a single-line expression")
        text = first_nonempty[0]
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    err = _validate_ast(tree)
    if err:
        fail(err)
    nodes = _count_nodes(tree)
    if nodes > MAX_NODES:
        fail("program too large (%d nodes)" % nodes)
    if nodes == 0:
        fail("empty expression")
    try:
        code = compile(tree, "<expr>", "eval")
    except Exception:
        fail("compile error")
    return code, nodes


def eval_row(code, r, t):
    env = dict(ALLOWED_FUNCS)
    env["r"] = r
    env["t"] = t
    try:
        v = eval(code, {"__builtins__": {}}, env)
    except ZeroDivisionError:
        fail("division by zero")
    except OverflowError:
        fail("numeric overflow")
    except _DomainError:
        fail("domain error (e.g. sqrt of negative)")
    except Exception:
        fail("evaluation error")
    if isinstance(v, complex):
        fail("complex result")
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        fail("non-numeric result")
    v = float(v)
    if v != v or v in (float("inf"), float("-inf")):
        fail("non-finite result")
    if abs(v) > 1e9:
        fail("result out of range")
    return v


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            header = fh.readline().split()
        cid = int(header[1])
    except Exception:
        fail("bad instance header")
    if cid < 1 or cid > 100000:
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

    A, Bc, c1, c2, c3 = hidden_params(cid)
    rows = heldout_sample(cid, A, Bc, c1, c2, c3)

    se = 0.0
    sb = 0.0
    for r, t, s_obs in rows:
        p = eval_row(code, r, t)
        se += (p - s_obs) ** 2
        sb += (BASELINE_CONST - s_obs) ** 2
    F_mse = se / len(rows)
    B_mse = sb / len(rows)

    B = B_mse * (1.0 + LAMBDA * 1)
    F = F_mse * (1.0 + LAMBDA * nodes)
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("heldout_MSE=%.6f baseline_MSE=%.6f nodes=%d  Ratio: %.6f"
          % (F_mse, B_mse, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
