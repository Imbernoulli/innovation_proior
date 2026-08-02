#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the two-mode dispersion-curve-invert task.

- Reads the test id `t` from <in>'s header (the rest of <in> is only used to
  cross-check the header; the hidden law is regenerated from `t` alone,
  IDENTICAL to gen.py's params()/true_v()).
- Parses the participant's closed-form velocity expression `v(f)` (single
  arithmetic expression over the variable `f`, using +,-,*,/,**, parentheses,
  numeric constants, and the unary functions sqrt/abs, plus the BINARY
  functions min/max).
- Evaluates it on a HELD-OUT frequency band that starts strictly above the
  training band and spans the mode-crossing region for most test ids, against
  the TRUE (noisy-measurement) held-out trace, and scores from mean pointwise
  accuracy (exp-decay of relative error) times a parsimony factor:
      F = mean_i exp(-relerr_i / REF) / (1 + LAMBDA * nodes)
      B = (same formula) for the checker's own constant-mean-of-train baseline
      Ratio = min(1000, 100*F/B) / 1000
  A constant reproduces ~0.1. A single-branch curve fit that ignores the
  crossing degrades sharply past f_cross. Sensor noise on both splits keeps
  even a perfect two-branch fit below the ceiling (headroom).
"""
import sys, math, ast, random

REF = 0.20
LAMBDA = 0.01
MAX_NODES = 40
MAX_OUT_BYTES = 20000

N_TRAIN = 26
F_TRAIN_LO, F_TRAIN_HI = 3.0, 25.0
N_HELD = 40
F_HELD_LO, F_HELD_HI = 26.0, 90.0

REGIME_BUCKET = {
    1: (30.0, 82.0), 2: (30.0, 82.0), 3: (30.0, 82.0), 4: (30.0, 82.0),
    5: (30.0, 82.0), 6: (30.0, 82.0), 7: (30.0, 82.0),
    8: (9.0, 24.0), 9: (9.0, 24.0),
    0: (2.0, 7.0),
}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden dispersion law (byte-identical to gen.py) ----------
def branch_consts(rng):
    while True:
        CA = rng.uniform(0.8, 1.3)
        CB = rng.uniform(3.0, 6.0)
        CD = rng.uniform(5.0, 20.0)
        CE = rng.uniform(1.0, 4.0)
        if CB - CD / CE > 0.3:
            return CA, CB, CD, CE


def find_crossing(CA, CB, CD, CE):
    def g(f):
        return CA * math.sqrt(f) - (CB - CD / (f + CE))
    lo, hi = 1e-6, 5000.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if g(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def params(t):
    bucket = REGIME_BUCKET.get(t % 10, (35.0, 260.0))
    lo_t, hi_t = bucket
    rng = random.Random(9173 + t * 7919)
    k = CA = CB = CD = CE = fc = None
    for _ in range(50000):
        k = rng.uniform(1.0, 5.0)
        CA, CB, CD, CE = branch_consts(rng)
        fc = find_crossing(CA, CB, CD, CE)
        if lo_t <= fc <= hi_t:
            break
    return k, CA, CB, CD, CE, fc


def true_v(f, k, CA, CB, CD, CE):
    vA = CA * math.sqrt(k * f)
    vB = math.sqrt(k) * (CB - CD / (f + CE))
    return min(vA, vB)


def train_freqs():
    return [F_TRAIN_LO + i * (F_TRAIN_HI - F_TRAIN_LO) / (N_TRAIN - 1) for i in range(N_TRAIN)]


def held_freqs():
    r = F_HELD_HI / F_HELD_LO
    return [F_HELD_LO * (r ** (i / (N_HELD - 1))) for i in range(N_HELD)]


# ---------- expression DSL ----------
def _safe_sqrt(x):
    if x < 0.0:
        raise ValueError("sqrt of negative")
    return math.sqrt(x)


ALLOWED_UNARY = {"sqrt": _safe_sqrt, "abs": abs}
ALLOWED_BINARY = {"min": min, "max": max}

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
)


def _validate(tree):
    func_name_ids = set()  # id() of Name nodes that are Call targets (not variable refs)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func_name_ids.add(id(node.func))
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return "disallowed syntax %s" % type(node).__name__
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                return "disallowed call target"
            fn = node.func.id
            if fn in ALLOWED_UNARY:
                if node.keywords or len(node.args) != 1:
                    return "bad arity for %s" % fn
            elif fn in ALLOWED_BINARY:
                if node.keywords or len(node.args) != 2:
                    return "bad arity for %s" % fn
            else:
                return "unknown function %s" % fn
        if isinstance(node, ast.Name) and id(node) not in func_name_ids:
            if node.id != "f":
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


class _SafeEval(ast.NodeVisitor):
    def __init__(self, f):
        self.env = {"f": f}

    def visit(self, node):
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            return self.env[node.id]
        if isinstance(node, ast.UnaryOp):
            v = self.visit(node.operand)
            if isinstance(node.op, ast.USub):
                return -v
            if isinstance(node.op, ast.UAdd):
                return v
            raise ValueError("bad unary op")
        if isinstance(node, ast.BinOp):
            a = self.visit(node.left)
            b = self.visit(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                return a / b
            if isinstance(node.op, ast.Pow):
                if abs(b) > 8 or (a < 0 and abs(b - round(b)) > 1e-9):
                    raise ValueError("unsafe power")
                return a ** b
            raise ValueError("bad binop")
        if isinstance(node, ast.Call):
            fn = node.func.id
            if fn in ALLOWED_UNARY:
                return ALLOWED_UNARY[fn](self.visit(node.args[0]))
            if fn in ALLOWED_BINARY:
                return ALLOWED_BINARY[fn](self.visit(node.args[0]), self.visit(node.args[1]))
            raise ValueError("bad call")
        raise ValueError("bad node")


def parse_expr(text):
    text = text.strip()
    if not text:
        fail("empty expression")
    if len(text) > 4000:
        fail("expression too long")
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
    return tree, nodes


def eval_expr(tree, f):
    try:
        v = _SafeEval(f).visit(tree)
    except ZeroDivisionError:
        fail("division by zero")
    except Exception as e:
        fail("evaluation error: %s" % e)
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        fail("non-numeric result")
    v = float(v)
    if v != v or v in (float("inf"), float("-inf")):
        fail("non-finite result")
    return v


def accuracy(preds, truth):
    accs = []
    for p, y in zip(preds, truth):
        rel = abs(p - y) / max(abs(y), 1e-6)
        accs.append(math.exp(-rel / REF))
    return sum(accs) / len(accs)


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

    k, CA, CB, CD, CE, fc = params(t)

    # held-out truth: same law, independent noise realisation (irreducible floor)
    sigma_rel_held = 0.02
    rng = random.Random(87178291199 + t * 104729)
    hfreqs = held_freqs()
    truth = []
    for f in hfreqs:
        v = true_v(f, k, CA, CB, CD, CE)
        truth.append(v + rng.gauss(0.0, sigma_rel_held * max(v, 0.5)))

    preds = [eval_expr(tree, f) for f in hfreqs]

    F_raw = accuracy(preds, truth)
    F = F_raw / (1.0 + LAMBDA * nodes)

    # internal baseline: constant = mean of the (regenerated) training trace
    train_rng = random.Random(555013 + t * 131071)
    tfreqs = train_freqs()
    train_vals = []
    for f in tfreqs:
        v = true_v(f, k, CA, CB, CD, CE)
        train_vals.append(v + train_rng.gauss(0.0, (0.03 + 0.01 * ((t - 1) % 5)) * max(v, 0.5)))
    const = sum(train_vals) / len(train_vals)
    B_raw = accuracy([const] * len(hfreqs), truth)
    B = B_raw / (1.0 + LAMBDA * 1)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F_raw=%.6f B_raw=%.6f nodes=%d fcross=%.3f  Ratio: %.6f"
          % (F_raw, B_raw, nodes, fc, sc / 1000.0))


if __name__ == "__main__":
    main()
