#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for "find the conserved quantity of a hidden boost map".

- Reads the test id t from <in>'s header, then re-derives the hidden law
  (p1..p6, identical formula to gen.py) and regenerates a HELD-OUT, LARGE-RADIUS
  ring of transitions -- genuine extrapolation beyond the training ring -- from
  t alone.  The hidden law lives ONLY here (and in gen.py); it is never printed.
- Parses the participant's candidate invariant: a LINEAR combination over the
  declared symbolic feature library {X, Y, XX, XY, YY} (degree<=2 monomials),
  e.g. "XX - YY", "0.5*X - 2*XY + Y", "-YY + 3.0*X".  Only `+ - *` with a
  numeric-constant * library-name pattern are allowed (no feature*feature, no
  division, no calls) -- so the participant is choosing a COEFFICIENT VECTOR
  over the library, not an arbitrary curve.
- Feasibility: unknown names, feature*feature / division / calls, non-finite
  or oversized constants, program too large, or an expression that is (near-)
  CONSTANT across the given TRAIN states (the cheap "conserved by definition"
  exploit) all score 0.
- Score: evaluate I on every held-out TRANSITION (state, next_state); the
  candidate should leave I UNCHANGED by the transition.  Relative error
      rel = |I(next) - I(state)| / (1 + |I(state)|)
  smoothly converted to per-transition credit acc = TOL^2/(TOL^2+rel^2),
  averaged, then a small parsimony tax on expression size:
      F = mean(acc) - LAMBDA * max(0, nodes - FREE_NODES)
      B = same F for the checker's own fixed baseline candidate "XY"
      Ratio = min(1000, 100*F/B) / 1000
  A constant-across-the-sample cheat is rejected outright (feasibility gate,
  not scored via F), so it cannot exploit "unchanged by definition".
"""
import sys, math, ast, random

TCAP = 0.6
TOL = 0.7
LAMBDA = 0.02
FREE_TERMS = 1
MAX_NODES = 40
MAX_OUT_BYTES = 20000
COEF_MAX = 1.0e6
DEGEN_VAR_THRESH = 1.0e-4
N_HELD = 600
HELD_RLO, HELD_RHI = 3.0, 6.0

FEATURES = ("X", "Y", "XX", "XY", "YY")


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden law (identical to gen.py) ----------
def hidden_params(t):
    rng = random.Random(900011 + t * 7919)
    p1 = rng.uniform(-1.0, 1.0)
    p2 = rng.uniform(-1.0, 1.0)
    p3 = rng.uniform(0.3, 0.8)
    p4 = rng.uniform(0.5, 2.0)
    p5 = rng.uniform(0.5, 2.0)
    p6 = rng.uniform(-0.5, 0.5)
    return (p1, p2, p3, p4, p5, p6)


def step(x, y, params):
    p1, p2, p3, p4, p5, p6 = params
    g = p1 * x + p2 * y + p3 * math.sin(p4 * x + p5 * y) + p6 * (x * y)
    u = TCAP * math.tanh(g)
    c = (1.0 + u * u) / (1.0 - u * u)
    s = 2.0 * u / (1.0 - u * u)
    xp = c * x + s * y
    yp = s * x + c * y
    return xp, yp


def held_out_ring(t, n):
    """Held-out EXTRAPOLATION ring (large radius); regenerated here only."""
    rng = random.Random(715827883 + t * 15485863)
    params = hidden_params(t)
    rows = []
    for _ in range(n):
        r = rng.uniform(HELD_RLO, HELD_RHI)
        th = rng.uniform(0.0, 2.0 * math.pi)
        x = r * math.cos(th)
        y = r * math.sin(th)
        xp, yp = step(x, y, params)
        rows.append((x, y, xp, yp))
    return rows


# ---------- feature library ----------
def feat_vals(x, y):
    return {"X": x, "Y": y, "XX": x * x, "XY": x * y, "YY": y * y}


# ---------- expression parsing / validation ----------
def _is_constish(node):
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _is_constish(node.operand)
    return False


_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.USub, ast.UAdd,
)


def parse_expr(text):
    text = text.strip()
    if not text:
        fail("empty expression")
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            fail("disallowed syntax %s" % type(node).__name__)
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Mult):
                left_c, right_c = _is_constish(node.left), _is_constish(node.right)
                left_n = isinstance(node.left, ast.Name) and node.left.id in FEATURES
                right_n = isinstance(node.right, ast.Name) and node.right.id in FEATURES
                ok = (left_c and right_n) or (right_c and left_n)
                if not ok:
                    fail("multiplication must be coefficient*feature (no feature*feature)")
            elif not isinstance(node.op, (ast.Add, ast.Sub)):
                fail("only + - * of coefficient*feature terms allowed")
        if isinstance(node, ast.Name) and node.id not in FEATURES:
            fail("unknown name %s (library is X,Y,XX,XY,YY)" % node.id)
        if isinstance(node, ast.Constant):
            v = node.value
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                fail("non-numeric constant")
            v = float(v)
            if v != v or v in (float("inf"), float("-inf")):
                fail("non-finite constant")
            if abs(v) > COEF_MAX:
                fail("constant magnitude too large")
    size = sum(1 for nd in ast.walk(tree)
               if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Name, ast.Constant)))
    if size > MAX_NODES:
        fail("expression too large (%d nodes)" % size)
    # The parsimony tax is charged per LIBRARY TERM used (how many features
    # you touched), not per AST token -- so spelling a coefficient out
    # explicitly ("1.000000*XX") costs the same as a bare feature name.
    terms = sum(1 for nd in ast.walk(tree) if isinstance(nd, ast.Name))
    try:
        code = compile(tree, "<expr>", "eval")
    except Exception:
        fail("compile error")
    return code, terms


def eval_code(code, x, y):
    env = dict(feat_vals(x, y))
    try:
        v = eval(code, {"__builtins__": {}}, env)
    except Exception:
        fail("evaluation error")
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        fail("non-numeric result")
    v = float(v)
    if v != v or v in (float("inf"), float("-inf")):
        fail("non-finite result")
    return v


def score_code(code, terms, heldout):
    accs = []
    for (x, y, xp, yp) in heldout:
        a = eval_code(code, x, y)
        b = eval_code(code, xp, yp)
        rel = abs(b - a) / (1.0 + abs(a))
        accs.append((TOL * TOL) / (TOL * TOL + rel * rel))
    raw = sum(accs) / len(accs)
    return max(0.0, raw - LAMBDA * max(0, terms - FREE_TERMS))


BASELINE_CODE = compile(ast.parse("XY", mode="eval"), "<baseline>", "eval")


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            header = fh.readline().split()
            n_train = int(header[0])
            t = int(header[1])
            train_states = []
            for _ in range(n_train):
                parts = fh.readline().split()
                x, y = float(parts[0]), float(parts[1])
                train_states.append((x, y))
    except Exception:
        fail("bad instance file")
    if t < 1 or t > 100000 or n_train <= 0:
        fail("bad test id")

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace").strip()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) != 1:
        fail("output must be exactly one non-empty line: a single expression")

    code, terms = parse_expr(lines[0])

    # feasibility: reject an expression that is (near-)constant across the
    # given TRAIN states -- "conserved by definition" is not a discovery.
    vals = [eval_code(code, x, y) for (x, y) in train_states]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    if var < DEGEN_VAR_THRESH:
        fail("expression is (near-)constant across the train states -- degenerate")

    heldout = held_out_ring(t, N_HELD)
    F = score_code(code, terms, heldout)
    B = score_code(BASELINE_CODE, 1, heldout)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("terms=%d F=%.6f baseline=%.6f  Ratio: %.6f" % (terms, F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
