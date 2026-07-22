#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for "coffee percolating through sieves of many sizes".

- Reads the test id from <in> (header), then regenerates the hidden crossing
  law Pi_true(p, L) = sigmoid((p - pc) * L**(1/nu)) for this id (pc, nu live
  ONLY here and in gen.py -- never printed to the solver) and builds a HELD-OUT
  Monte-Carlo trace on much LARGER sieves L in {128, 512}, on a FINE p-grid
  concentrated near pc (a genuine extrapolation in L; L=128/512 never appear
  in training).
- Parses the participant's closed-form expression: arithmetic over the two
  variables `p`, `L`, numeric constants, + - * /, the unary functions sig,
  tanh, absv, sqrt, and the ONE two-argument function `pw(base, exponent)`
  (a safe real power, needed to express an L**(1/nu)-style term).
- Evaluates the expression on every held-out (p, L) point, forms the mean
  squared error against the true (noisy) crossing fraction, and scores
  (minimisation) with a small parsimony penalty on expression size:
      F = MSE * (1 + LAMBDA * nodes)
      B = MSE_of_constant_0.5 * (1 + LAMBDA * 1)     # internal baseline
      Ratio = min(1000, 100*B/F) / 1000
  A constant 0.5 reproduces the baseline (~0.1).  Any expression that gets
  the threshold LOCATION and the transition WIDTH right at both extrapolated
  sizes drives MSE down; Monte-Carlo noise plus the parsimony tax keep even a
  perfect law-recovery well below the ceiling, leaving headroom.
"""
import sys, math, ast, random

LAMBDA = 0.01
MAX_NODES = 40
MAX_OUT_BYTES = 20000
OFF, AMP = 0.1, 0.8
HELDOUT_LS = [128, 512]
N_P_HELD = 25
HELDOUT_SIGMA = 0.15
HELDOUT_HALFWIDTH_MULT = 5.5

ALLOWED_UNARY = {
    "sig": lambda x: _sig(x),
    "tanh": math.tanh,
    "absv": abs,
    "sqrt": lambda x: math.sqrt(abs(x)),
}


def _sig(x):
    x = max(-60.0, min(60.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _pw(base, exp):
    """Safe real power: base**exp, guarding non-positive base / overflow."""
    try:
        b = float(base)
        e = float(exp)
    except Exception:
        return float("nan")
    if b != b or e != e or abs(e) > 40.0:
        return float("nan")
    if b <= 1e-12:
        return 0.0
    try:
        v = math.exp(e * math.log(b))
    except OverflowError:
        return float("nan")
    return v


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden crossing law (identical to gen.py) ----------
def hidden_params(t):
    rng = random.Random(9013 + t * 7919)
    pc = rng.uniform(0.30, 0.70)
    nu = rng.uniform(0.9, 2.2)
    return pc, nu


def pi_true(p, L, pc, nu):
    x = (p - pc) * (L ** (1.0 / nu))
    return OFF + AMP * _sig(x)


def heldout_points(t, pc, nu):
    """Fine p-grid centred on pc, width scaled to the true transition width at
    each held-out L (a genuine test of both threshold LOCATION and SHARPNESS
    at sizes never seen in training). Each reading carries independent
    measurement noise, so even exact recovery of (pc, nu) leaves a nonzero
    residual -- the irreducible headroom above any strong solution."""
    rng = random.Random(31337 + t * 15485863)
    rows = []
    for L in HELDOUT_LS:
        halfwidth = max(0.0015, HELDOUT_HALFWIDTH_MULT * (L ** (-1.0 / nu)))
        halfwidth = min(halfwidth, 0.45)
        for i in range(N_P_HELD):
            frac = -1.0 + 2.0 * i / (N_P_HELD - 1)
            p = pc + frac * halfwidth
            p = min(1.0, max(0.0, p))
            true_pi = pi_true(p, L, pc, nu)
            obs = true_pi + rng.gauss(0.0, HELDOUT_SIGMA)
            obs = min(1.0, max(0.0, obs))
            rows.append((L, p, obs))
    return rows


# ---------- expression parsing / validation ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
)
_ALLOWED_CALL_NAMES = set(ALLOWED_UNARY) | {"pw"}


def _validate_ast(tree):
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            fail("disallowed syntax %s" % type(node).__name__)
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_CALL_NAMES):
                fail("disallowed call")
            if node.keywords:
                fail("bad function arity")
            nargs = 2 if node.func.id == "pw" else 1
            if len(node.args) != nargs:
                fail("bad function arity for %s" % node.func.id)
        if isinstance(node, ast.Name):
            if node.id in _ALLOWED_CALL_NAMES:
                continue
            if node.id not in ("p", "L"):
                fail("unknown name %s" % node.id)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                fail("non-numeric constant")
            v = float(node.value)
            if v != v or v in (float("inf"), float("-inf")):
                fail("non-finite constant")


def _count_nodes(tree):
    return sum(1 for nd in ast.walk(tree)
               if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant)))


def parse_expr(raw):
    text = raw.strip()
    if not text:
        fail("empty output")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        fail("empty output")
    text = lines[0].strip()
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    _validate_ast(tree)
    nodes = _count_nodes(tree)
    if nodes > MAX_NODES:
        fail("expression too large (%d nodes)" % nodes)
    if nodes == 0:
        fail("empty expression")
    return tree, nodes


def _eval_node(node, p, L):
    """Direct recursive evaluator (no builtins reachable) -- avoids relying on
    Python's eval() semantics for exponent-like tricks; every call routes
    through the vetted ALLOWED_UNARY / _pw table."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, p, L)
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id == "p":
            return p
        if node.id == "L":
            return L
        fail("unknown name %s" % node.id)
    if isinstance(node, ast.UnaryOp):
        v = _eval_node(node.operand, p, L)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return v
        fail("bad unary op")
    if isinstance(node, ast.BinOp):
        a = _eval_node(node.left, p, L)
        b = _eval_node(node.right, p, L)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Div):
            if abs(b) < 1e-12:
                return float("nan")
            v = a / b
            if abs(v) > 1e12:
                return float("nan")
            return v
        fail("bad binary op")
    if isinstance(node, ast.Call):
        name = node.func.id
        if name == "pw":
            base = _eval_node(node.args[0], p, L)
            exp = _eval_node(node.args[1], p, L)
            return _pw(base, exp)
        fn = ALLOWED_UNARY[name]
        arg = _eval_node(node.args[0], p, L)
        return fn(arg)
    fail("bad node")


def eval_expr(tree_or_code, p, L):
    return _eval_node(tree_or_code, p, L)


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

    tree, nodes = parse_expr(text)

    pc, nu = hidden_params(t)
    rows = heldout_points(t, pc, nu)

    se = 0.0
    for L, p, pih in rows:
        try:
            pred = eval_expr(tree, float(p), float(L))
        except SystemExit:
            raise
        except Exception:
            fail("evaluation error")
        if not isinstance(pred, (int, float)) or isinstance(pred, bool):
            fail("non-numeric result")
        pred = float(pred)
        if pred != pred or pred in (float("inf"), float("-inf")):
            fail("non-finite result")
        if abs(pred) > 1e6:
            fail("prediction magnitude out of range")
        try:
            se += (pred - pih) ** 2
        except OverflowError:
            fail("overflow in scoring")
    F_mse = se / len(rows)
    B_mse = sum((0.5 - pih) ** 2 for _, _, pih in rows) / len(rows)

    B = B_mse * (1.0 + LAMBDA * 1)
    F = F_mse * (1.0 + LAMBDA * nodes)
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("heldout_MSE=%.6f baseline_MSE=%.6f nodes=%d  Ratio: %.6f"
          % (F_mse, B_mse, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
