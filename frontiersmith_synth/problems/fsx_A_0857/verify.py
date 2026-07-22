#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic scorer for the merge-junction congestion-law EXTRAPOLATION
problem (format E).

- <in>  : the LIGHT-traffic train rows the solver saw (regenerated
          identically here to recover the test_id).
- <out> : the solver's submitted closed-form expression in x0, x1, x2.

The checker regenerates the hidden law (shared congestion exponent k, the
per-feeder coupling weights w1/w2, scale b, capacity c) and a held-out
HEAVY-traffic region -- a mix of deterministic "trap" corner
points (own-flow-dominant vs. coupling-dominant combinations) plus randomly
sampled points, all strictly outside the light-traffic training box.  It
evaluates the submitted expression there and scores extrapolation RMSE (with
a small complexity penalty) against an internal constant-predictor baseline.

Minimization objective:
    F  = RMSE_heldout + ALPHA * complexity
    B  = RMSE of the constant predictor (mean of train y) on the held-out set
    sc = min(1000, 100 * B / max(eps, F));  Ratio = sc / 1000

Any feasibility violation (unparseable / disallowed / non-finite / absurd
output) prints Ratio: 0.0 and exits 0.
"""
import sys
import ast
import math

ALPHA = 0.004
MAX_CHARS = 5000
MAX_NODES = 400

ALLOWED_FUNCS = {
    "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
    "sin": math.sin, "cos": math.cos, "tanh": math.tanh, "abs": abs,
}
ALLOWED_VARS = {"x0", "x1", "x2"}
ALLOWED_CONSTS = {"pi": math.pi, "e": math.e}


# ============ hidden ground truth (mirrors gen.py exactly) ============
def _rng(seed):
    state = [(seed * 2654435761 + 12345) & 0x7FFFFFFF]

    def nxt():
        state[0] = (1103515245 * state[0] + 12345) & 0x7FFFFFFF
        return state[0] / 0x7FFFFFFF

    return nxt


def derive_params(test_id):
    r = _rng(1000 + test_id)
    k = 1.5 + 1.2 * r()
    c = 8.0 + 4.0 * r()
    b = 2.5 + 2.0 * r()
    w1 = 0.3 + 0.4 * r()
    w2 = 0.2 + 0.4 * r()
    return k, c, b, w1, w2


def n_train(test_id):
    return 60 + 6 * (test_id - 1)


def noise_rel(test_id):
    return 0.18 + 0.02 * (test_id - 1)


LOW_LO, LOW_HI = 0.10, 3.0
HIGH_LO, HIGH_HI = 4.0, 7.0
HIGH_MID = 0.5 * (HIGH_LO + HIGH_HI)
N_HELD_RANDOM = 14


def clean_y(x0, x1, x2, params):
    k, c, b, w1, w2 = params
    u = x0 + w1 * x1 + w2 * x2
    return b * (u / c) ** k


def make_train_rows(test_id):
    params = derive_params(test_id)
    n = n_train(test_id)
    nr = noise_rel(test_id)
    rx = _rng(3000 + test_id)
    rn = _rng(5000 + test_id)
    rows = []
    for _ in range(n):
        x0 = LOW_LO + (LOW_HI - LOW_LO) * rx()
        x1 = LOW_LO + (LOW_HI - LOW_LO) * rx()
        x2 = LOW_LO + (LOW_HI - LOW_LO) * rx()
        clean = clean_y(x0, x1, x2, params)
        y = clean * (1.0 + nr * (2.0 * rn() - 1.0))
        rows.append((x0, x1, x2, y))
    return rows


def held_points(test_id):
    """HEAVY-traffic held-out region: deterministic trap corners covering
    own-flow-dominant vs. coupling(feeder)-dominant combinations, plus
    randomly sampled points -- all strictly outside the training box."""
    pts = [
        (HIGH_LO, HIGH_HI, HIGH_HI),   # coupling-dominant: light own flow, both feeders heavy
        (HIGH_HI, HIGH_LO, HIGH_LO),   # own-dominant: heavy own flow, feeders light
        (HIGH_LO, HIGH_LO, HIGH_HI),   # light own flow, only feeder 2 heavy
        (HIGH_HI, HIGH_HI, HIGH_LO),   # heavy own flow + feeder 1 heavy
        (HIGH_MID, HIGH_MID, HIGH_MID),  # balanced
        (HIGH_LO, HIGH_HI, HIGH_LO),   # light own flow, only feeder 1 heavy
    ]
    rx = _rng(7000 + test_id)
    for _ in range(N_HELD_RANDOM):
        x0 = HIGH_LO + (HIGH_HI - HIGH_LO) * rx()
        x1 = HIGH_LO + (HIGH_HI - HIGH_LO) * rx()
        x2 = HIGH_LO + (HIGH_HI - HIGH_LO) * rx()
        pts.append((x0, x1, x2))
    return pts


def make_held(test_id):
    params = derive_params(test_id)
    nr = noise_rel(test_id)
    rn = _rng(9000 + test_id)   # separate held-out noise stream
    pts = held_points(test_id)
    ys = []
    for x0, x1, x2 in pts:
        clean = clean_y(x0, x1, x2, params)
        ys.append(clean * (1.0 + nr * (2.0 * rn() - 1.0)))
    return pts, ys


# ============ safe expression evaluation (whitelist AST) ============
def _check_node(node):
    if isinstance(node, ast.Expression):
        _check_node(node.body)
    elif isinstance(node, ast.BinOp):
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult,
                                    ast.Div, ast.Pow, ast.Mod)):
            raise ValueError("bad binop")
        _check_node(node.left)
        _check_node(node.right)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise ValueError("bad unaryop")
        _check_node(node.operand)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCS:
            raise ValueError("bad call")
        if node.keywords or len(node.args) != 1:
            raise ValueError("bad call args")
        _check_node(node.args[0])
    elif isinstance(node, ast.Name):
        if node.id not in ALLOWED_VARS and node.id not in ALLOWED_CONSTS:
            raise ValueError("bad name: %s" % node.id)
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
            raise ValueError("bad const")
        if not math.isfinite(float(node.value)):
            raise ValueError("nonfinite const")
    else:
        raise ValueError("disallowed node: %s" % type(node).__name__)


def _count_nodes(node):
    return sum(1 for _ in ast.walk(node))


def compile_expr(text):
    if len(text) > MAX_CHARS:
        raise ValueError("expr too long")
    if "\n" in text:
        raise ValueError("expr must be a single line")
    tree = ast.parse(text, mode="eval")
    _check_node(tree)
    complexity = _count_nodes(tree.body)
    if complexity > MAX_NODES:
        raise ValueError("too complex")
    code = compile(tree, "<expr>", "eval")

    def ev(x0, x1, x2):
        env = {"x0": x0, "x1": x1, "x2": x2}
        env.update(ALLOWED_CONSTS)
        env.update(ALLOWED_FUNCS)
        return eval(code, {"__builtins__": {}}, env)

    return ev, complexity


def fail(reason):
    print("reason: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def rmse(pred, targ):
    s = 0.0
    for p, t in zip(pred, targ):
        d = p - t
        s += d * d
    return math.sqrt(s / len(targ))


def main():
    if len(sys.argv) < 3:
        fail("usage")
    in_path, out_path = sys.argv[1], sys.argv[2]

    try:
        toks = open(in_path).read().split()
        it = iter(toks)
        n = int(next(it))
        obs_rows = []
        for _ in range(n):
            x0 = float(next(it)); x1 = float(next(it))
            x2 = float(next(it)); y = float(next(it))
            obs_rows.append((x0, x1, x2, y))
    except Exception:
        fail("bad instance")

    # identify test_id by matching the regenerated train rows (robust: the
    # full row content -- x's and noisy y's -- encodes seed + params).
    test_id = None
    for tid in range(1, 300):
        if n_train(tid) != len(obs_rows):
            continue
        gen_rows = make_train_rows(tid)
        ok = True
        for (gx0, gx1, gx2, gy), (ox0, ox1, ox2, oy) in zip(gen_rows, obs_rows):
            if (abs(gx0 - ox0) > 1e-6 * (1 + abs(gx0)) or
                    abs(gx1 - ox1) > 1e-6 * (1 + abs(gx1)) or
                    abs(gx2 - ox2) > 1e-6 * (1 + abs(gx2)) or
                    abs(gy - oy) > 1e-6 * (1 + abs(gy))):
                ok = False
                break
        if ok:
            test_id = tid
            break
    if test_id is None:
        fail("unrecognized instance")

    try:
        blob = open(out_path).read()
    except Exception:
        fail("no output")
    lines = [ln.strip() for ln in blob.splitlines() if ln.strip()]
    if not lines:
        fail("empty output")
    expr = lines[-1]
    if expr.count("=") == 1 and "==" not in expr:
        expr = expr.split("=", 1)[1].strip()

    try:
        ev, complexity = compile_expr(expr)
    except Exception as e:
        fail("unparseable/disallowed: %s" % e)

    hpts, hy = make_held(test_id)
    preds = []
    for x0, x1, x2 in hpts:
        try:
            v = ev(x0, x1, x2)
        except Exception as e:
            fail("eval error: %s" % e)
        v = float(v)
        if not math.isfinite(v):
            fail("non-finite prediction")
        if abs(v) > 1e6:
            fail("absurd prediction magnitude")
        preds.append(v)

    err = rmse(preds, hy)
    eff = err * (1.0 + ALPHA * complexity)

    ty = [r[3] for r in make_train_rows(test_id)]
    tmean = sum(ty) / len(ty)
    B = rmse([tmean] * len(hy), hy)

    sc = min(1000.0, 100.0 * B / max(1e-9, eff))
    print("test_id=%d B=%.6g err=%.6g complexity=%d eff=%.6g" %
          (test_id, B, err, complexity, eff))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
