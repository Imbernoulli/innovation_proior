#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic scorer for the hidden-response-law-recovery EXTRAPOLATION
problem (format E).

- <in>  : the small-signal / sub-resonant train rows the solver saw
          (regenerated identically here to recover the test_id).
- <out> : the solver's submitted closed-form expression in f, a.

The checker regenerates the hidden law
    R(f,a) = G * (As*tanh(a/As)) / (1 + ((f-f0)/w)**2)
(overall gain G, saturation scale As, resonance frequency f0, resonance
half-width w) and a held-out region that is genuinely disjoint from the
training box on BOTH axes: large-signal amplitudes (deep into saturation)
crossed with frequencies at/beyond the resonance peak (through the rolloff)
-- including deterministic "trap" corners that isolate the saturation
effect from the resonance effect and combine both. It evaluates the
submitted expression there and scores extrapolation RMSE (with a small
complexity penalty) against an internal constant-predictor baseline.

Minimization objective:
    err = RMSE_heldout
    eff = err * (1 + ALPHA * complexity)
    B   = RMSE of the constant predictor (mean of train y) on the held-out set
    sc  = min(1000, 100 * B / max(eps, eff));  Ratio = sc / 1000

Any feasibility violation (unparseable / disallowed / non-finite / absurd
output) prints Ratio: 0.0 and exits 0.
"""
import sys
import ast
import math

ALPHA = 0.003
MAX_CHARS = 5000
MAX_NODES = 400

ALLOWED_FUNCS = {
    "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
    "sin": math.sin, "cos": math.cos, "tanh": math.tanh, "abs": abs,
}
ALLOWED_VARS = {"f", "a"}
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
    G = 1.5 + 1.5 * r()
    As = 3.0 + 3.0 * r()
    f0 = 10.0 + 6.0 * r()
    w = 1.2 + 1.2 * r()
    return G, As, f0, w


def n_train(test_id):
    return 60 + 6 * (test_id - 1)


def noise_rel(test_id):
    return 0.05 + 0.01 * (test_id - 1)


TRAIN_F_LO = 0.1
TRAIN_F_MARGIN_W = 1.5
TRAIN_A_LO_FRAC = 0.05
TRAIN_A_HI_FRAC = 0.75

HELD_F_LO_MULT_W = 0.8    # held f low  = f0 - 0.8*w   (approaching the peak)
HELD_F_HI_MULT_W = 2.4    # held f high (hard corners) = f0 + 2.4*w  (past the peak / rolloff)
HELD_A_LO_MULT = 1.3      # held a low  = 1.3*As       (past small-signal, into saturation)
HELD_A_HI_MULT = 2.0      # held a high (hard corners) = 2.0*As      (deep saturation)
# the bulk of held points (random draws) come from a MODERATE extrapolation
# sub-box -- still strictly outside training, still requires generalization
# -- while a handful of deterministic corners probe the full severe box
# (through the peak, deep saturation). This keeps the ladder honest: a
# recipe that gets the general trend right should do respectably on the
# moderate majority, while still visibly failing the hard corners.
HELD_F_RAND_HI_MULT_W = 1.0
HELD_A_RAND_HI_MULT = 1.6
N_HELD_RANDOM = 15


def clean_R(f, a, params):
    G, As, f0, w = params
    sat = As * math.tanh(a / As)
    res = 1.0 / (1.0 + ((f - f0) / w) ** 2)
    return G * sat * res


def make_train_rows(test_id):
    params = derive_params(test_id)
    G, As, f0, w = params
    n = n_train(test_id)
    nr = noise_rel(test_id)
    f_hi = f0 - TRAIN_F_MARGIN_W * w
    a_lo = TRAIN_A_LO_FRAC * As
    a_hi = TRAIN_A_HI_FRAC * As
    rx = _rng(3000 + test_id)
    rn = _rng(5000 + test_id)
    rows = []
    for _ in range(n):
        f = TRAIN_F_LO + (f_hi - TRAIN_F_LO) * rx()
        a = a_lo + (a_hi - a_lo) * rx()
        clean = clean_R(f, a, params)
        y = clean * (1.0 + nr * (2.0 * rn() - 1.0))
        rows.append((f, a, y))
    return rows


def held_points(test_id, params):
    """Held-out region: deterministic trap corners covering
    resonance-only / saturation-only / both-combined extrapolation, plus
    randomly sampled points -- all strictly outside the training box on
    BOTH axes (see derivation: held_f_lo - train_f_hi = 0.7*w > 0 and
    held_a_lo - train_a_hi = 0.55*As > 0 for every valid draw)."""
    G, As, f0, w = params
    f_lo = f0 - HELD_F_LO_MULT_W * w
    f_hi = f0 + HELD_F_HI_MULT_W * w
    a_lo = HELD_A_LO_MULT * As
    a_hi = HELD_A_HI_MULT * As
    # training-box bounds (same formulas as make_train_rows) -- used to build
    # genuine SINGLE-AXIS isolation corners: one axis pinned INSIDE the
    # training box (interpolation) while the other is pushed into the held
    # region (extrapolation), isolating that one effect from the other.
    train_f_hi = f0 - TRAIN_F_MARGIN_W * w
    train_f_mid = 0.5 * (TRAIN_F_LO + train_f_hi)
    train_a_lo = TRAIN_A_LO_FRAC * As
    train_a_hi = TRAIN_A_HI_FRAC * As
    train_a_mid = 0.5 * (train_a_lo + train_a_hi)
    pts = [
        (f_lo, a_hi),           # near-peak approach, deep saturation (combined)
        (f0, a_hi),             # exact resonance peak, deep saturation (sharpest combined trap)
        (f_hi, a_lo),           # deep rolloff, edge of saturation (combined)
        (train_f_mid, a_hi),    # AMPLITUDE-ONLY: f inside training box, a deep in saturation
        (f_hi, train_a_mid),    # FREQUENCY-ONLY: f in deep rolloff, a inside training box
    ]
    # bulk random points: a MODERATE extrapolation sub-box (still strictly
    # outside training on both axes, still requires generalization)
    f_rand_hi = f0 + HELD_F_RAND_HI_MULT_W * w
    a_rand_hi = HELD_A_RAND_HI_MULT * As
    rx = _rng(7000 + test_id)
    for _ in range(N_HELD_RANDOM):
        f = f_lo + (f_rand_hi - f_lo) * rx()
        a = a_lo + (a_rand_hi - a_lo) * rx()
        pts.append((f, a))
    return pts


def make_held(test_id):
    params = derive_params(test_id)
    nr = noise_rel(test_id)
    rn = _rng(9000 + test_id)   # separate held-out noise stream
    pts = held_points(test_id, params)
    ys = []
    for f, a in pts:
        clean = clean_R(f, a, params)
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

    def ev(f, a):
        env = {"f": f, "a": a}
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
            f = float(next(it)); a = float(next(it)); y = float(next(it))
            obs_rows.append((f, a, y))
    except Exception:
        fail("bad instance")

    # identify test_id by matching the regenerated train rows (robust: the
    # full row content -- f, a and the noisy y's -- encodes seed + params).
    test_id = None
    for tid in range(1, 300):
        if n_train(tid) != len(obs_rows):
            continue
        gen_rows = make_train_rows(tid)
        good = True
        for (gf, ga, gy), (of, oa, oy) in zip(gen_rows, obs_rows):
            if (abs(gf - of) > 1e-6 * (1 + abs(gf)) or
                    abs(ga - oa) > 1e-6 * (1 + abs(ga)) or
                    abs(gy - oy) > 1e-6 * (1 + abs(gy))):
                good = False
                break
        if good:
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
    if len(lines) > 1:
        fail("multi-line output")
    expr = lines[0]

    try:
        ev, complexity = compile_expr(expr)
    except Exception as e:
        fail("unparseable/disallowed: %s" % e)

    hpts, hy = make_held(test_id)
    preds = []
    for f, a in hpts:
        try:
            v = ev(f, a)
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

    ty = [r[2] for r in make_train_rows(test_id)]
    tmean = sum(ty) / len(ty)
    B = rmse([tmean] * len(hy), hy)

    sc = min(1000.0, 100.0 * B / max(1e-9, eff))
    print("test_id=%d B=%.6g err=%.6g complexity=%d eff=%.6g" %
          (test_id, B, err, complexity, eff))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
