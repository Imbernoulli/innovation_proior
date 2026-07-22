#!/usr/bin/env python3
# verify.py -- deterministic scorer for the Brass-Tag Registry Number recovery (format E).
# CLI:  python3 verify.py <in> <out> <ans>     (ans is ignored)
# Prints exactly one final "Ratio: <float in [0,1]>" line and exits 0.
#
# Anti-leak: the hidden law + the withheld (100000..1000000) evaluation split are regenerated
# HERE (never shipped as an importable module, never printed by gen.py). The withheld region is
# genuine EXTRAPOLATION -- tag numbers 30x-300x beyond the training range -- so it rewards a
# solver that recovers the multiplicative, prime-power-indexed law, not one that memorises the
# training numbers or fits a smooth curve in n.
#
# Output contract for the participant (two lines):
#   line 1:  MODE N   -- OR --   MODE PP
#   line 2:  a closed-form Python expression string
#     * MODE N  : expression is a function of the single variable  n
#     * MODE PP : expression is a function of a PRIME p and an exponent k, i.e. of the prime
#                  POWER p**k.  The checker factors each withheld n into its prime-power parts
#                  p1**k1 * p2**k2 * ... (a fixed, disclosed evaluation schema -- exactly what
#                  "the domain decomposition the law respects" means here) and multiplies the
#                  submitted expression's value over those parts to get the predicted f(n).
#
# Objective (minimise): F = (clipped mean relative error on the withheld split + a floor
# constant that keeps irreducible headroom) * a gentle expression-size penalty.  Internal
# baseline B is the same functional but for the fixed trivial predictor f(n) = n.
import sys, math, ast, random

# ---- FIXED ladder (byte-identical to gen.py) ----
COEF_TABLE = {
    1:  (1, 1, 1),
    2:  (0, 2, 0),
    3:  (1, 0, 2),
    4:  (0, 1, 2),
    5:  (1, 2, 1),
    6:  (0, 1, 1),
    7:  (1, 0, 1),
    8:  (0, 2, 1),
    9:  (1, 1, 2),
    10: (1, 2, 0),
}
# N_train is the SAME for every testId on purpose -- see gen.py for why (no line-1 side channel).
NTRAIN_FIXED = 3000
SIGMA_TABLE = {1: 0.4, 2: 0.5, 3: 0.6, 4: 0.7, 5: 0.8,
               6: 1.0, 7: 1.1, 8: 1.3, 9: 1.5, 10: 1.8}

N_HOLD = 2200
HOLD_LO, HOLD_HI = 100_000, 1_000_000
REL_ERR_CAP = 5.0
FLOOR = 0.25
COMPLEXITY_FREE = 40
COMPLEXITY_RATE = 0.006
MAX_EXPR_LEN = 2000
MAX_POW_EXP = 12


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


def coef_for(test_id):
    a2, a1, a3 = COEF_TABLE[test_id]
    return a2, a1, a3


def pp_value(p, k, a2, a1, a3):
    if p == 2:
        a = a2
    elif p % 4 == 1:
        a = a1
    else:
        a = a3
    return p ** k - a * p ** (k - 1)


def factorize(n):
    """Trial-division factorization; n <= 1e6 so this is O(sqrt n)."""
    m = n
    d = 2
    factors = []
    while d * d <= m:
        if m % d == 0:
            k = 0
            while m % d == 0:
                m //= d
                k += 1
            factors.append((d, k))
        d += 1 if d == 2 else 2
    if m > 1:
        factors.append((m, 1))
    return factors


def f_true(n, a2, a1, a3):
    if n == 1:
        return 1
    result = 1
    for (p, k) in factorize(n):
        result *= pp_value(p, k, a2, a1, a3)
    return result


def gen_train(test_id):
    a2, a1, a3 = coef_for(test_id)
    ntrain = NTRAIN_FIXED
    sigma = SIGMA_TABLE[test_id]
    rng = random.Random(4_100_000 + test_id * 131)
    rows = []
    for n in range(1, ntrain + 1):
        true_v = f_true(n, a2, a1, a3)
        noise = int(round(rng.gauss(0.0, sigma)))
        noise = max(-5, min(5, noise))
        obs = true_v + noise
        if obs < 0:
            obs = 0
        rows.append((n, obs))
    return rows


def gen_holdout_ns(test_id):
    rng = random.Random(9_500_000 + test_id * 977)
    return [rng.randint(HOLD_LO, HOLD_HI) for _ in range(N_HOLD)]


# ---------------- safe expression evaluator (AST whitelist, no eval()) ----------------
_ALLOWED_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
                    ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
                    ast.Mod: lambda a, b: a % b}
_ALLOWED_FUNCS = {"abs": abs, "min": min, "max": max}


def validate_exponent(node, pp_mode):
    """A Pow exponent must be PROVABLY a small bounded integer at parse time: an int literal,
    the exponent variable 'k' (pp mode only, always a small positive int at runtime), or a
    +/- combination of those.  This forbids p**n / n**n style blow-ups statically -- no
    exponent subtree can ever evaluate outside a tiny, bounded range."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int) or abs(node.value) > MAX_POW_EXP:
            raise ValueError("pow exponent constant out of range")
        return
    if isinstance(node, ast.Name):
        if pp_mode and node.id == "k":
            return
        raise ValueError("pow exponent name %r not allowed" % node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        validate_exponent(node.operand, pp_mode); return
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        validate_exponent(node.left, pp_mode); validate_exponent(node.right, pp_mode); return
    raise ValueError("pow exponent must be a small bounded int expression")


def validate_ast(node, allowed_names, pp_mode):
    if isinstance(node, ast.Expression):
        validate_ast(node.body, allowed_names, pp_mode); return
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Pow):
            validate_ast(node.left, allowed_names, pp_mode)
            validate_exponent(node.right, pp_mode)
            return
        if type(node.op) not in _ALLOWED_BINOPS:
            raise ValueError("binop %s" % type(node.op).__name__)
        validate_ast(node.left, allowed_names, pp_mode)
        validate_ast(node.right, allowed_names, pp_mode)
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise ValueError("unary")
        validate_ast(node.operand, allowed_names, pp_mode); return
    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, (ast.And, ast.Or)):
            raise ValueError("boolop")
        for v in node.values:
            validate_ast(v, allowed_names, pp_mode)
        return
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or not isinstance(node.ops[0], (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
            raise ValueError("compare")
        validate_ast(node.left, allowed_names, pp_mode)
        validate_ast(node.comparators[0], allowed_names, pp_mode)
        return
    if isinstance(node, ast.IfExp):
        validate_ast(node.test, allowed_names, pp_mode)
        validate_ast(node.body, allowed_names, pp_mode)
        validate_ast(node.orelse, allowed_names, pp_mode)
        return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise ValueError("call")
        if node.keywords:
            raise ValueError("kwargs")
        if not (1 <= len(node.args) <= 2):
            raise ValueError("arity")
        for a in node.args:
            validate_ast(a, allowed_names, pp_mode)
        return
    if isinstance(node, ast.Name):
        if node.id not in allowed_names:
            raise ValueError("name %r" % node.id)
        return
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("const type")
        return
    raise ValueError("node %s" % type(node).__name__)


def eval_ast(node, env):
    if isinstance(node, ast.Expression):
        return eval_ast(node.body, env)
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Pow):
            base = eval_ast(node.left, env)
            exp = eval_ast(node.right, env)
            if abs(exp) > 1000:            # defense-in-depth; validate_exponent already bounds this
                raise ValueError("pow exponent too large at runtime")
            return base ** exp
        a = eval_ast(node.left, env); b = eval_ast(node.right, env)
        return _ALLOWED_BINOPS[type(node.op)](a, b)
    if isinstance(node, ast.UnaryOp):
        v = eval_ast(node.operand, env)
        return +v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.BoolOp):
        is_and = isinstance(node.op, ast.And)
        result = None
        for v in node.values:                    # real Python and/or: short-circuit, return
            result = eval_ast(v, env)             # the operand VALUE, not a bool
            if is_and and not result:
                return result
            if (not is_and) and result:
                return result
        return result
    if isinstance(node, ast.Compare):
        a = eval_ast(node.left, env); b = eval_ast(node.comparators[0], env)
        op = node.ops[0]
        if isinstance(op, ast.Eq): return a == b
        if isinstance(op, ast.NotEq): return a != b
        if isinstance(op, ast.Lt): return a < b
        if isinstance(op, ast.LtE): return a <= b
        if isinstance(op, ast.Gt): return a > b
        if isinstance(op, ast.GtE): return a >= b
    if isinstance(node, ast.IfExp):
        return eval_ast(node.body, env) if eval_ast(node.test, env) else eval_ast(node.orelse, env)
    if isinstance(node, ast.Call):
        args = [eval_ast(a, env) for a in node.args]
        return _ALLOWED_FUNCS[node.func.id](*args)
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.Constant):
        return node.value
    raise ValueError("eval node %s" % type(node).__name__)


def clipped_rel_err(pred, true_v):
    re = abs(pred - true_v) / true_v
    return min(REL_ERR_CAP, re)


def main():
    if len(sys.argv) < 3:
        fail("usage")
    in_path, out_path = sys.argv[1], sys.argv[2]

    # ---- read instance ----
    try:
        with open(in_path) as f:
            toks = f.read().split()
    except Exception:
        fail("no instance")
    if not toks:
        fail("empty instance")
    try:
        ntr = int(toks[0])
    except Exception:
        fail("bad Ntrain")
    if ntr <= 0 or len(toks) < 1 + 2 * ntr:
        fail("truncated instance")
    train = []
    idx = 1
    for _ in range(ntr):
        n = int(toks[idx]); obs = int(toks[idx + 1])
        idx += 2
        train.append((n, obs))

    # ---- re-identify which testId produced this instance ----
    test_id = None
    for tid in range(1, 11):
        cand = gen_train(tid)
        if len(cand) != len(train):
            continue
        if cand == train:
            test_id = tid
            break
    if test_id is None:
        fail("instance not recognised")

    # ---- read participant output ----
    try:
        with open(out_path) as f:
            raw = f.read()
    except Exception:
        fail("no output")
    if len(raw) > MAX_EXPR_LEN:
        fail("output too long")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip() != ""]
    if len(lines) != 2:
        fail("output must be exactly two non-blank lines: MODE line + expression line")
    mode_line, expr = lines[0], lines[1]
    mode_toks = mode_line.split()
    if len(mode_toks) != 2 or mode_toks[0].upper() != "MODE" or mode_toks[1].upper() not in ("N", "PP"):
        fail("bad MODE line (expected 'MODE N' or 'MODE PP')")
    mode = mode_toks[1].upper()
    pp_mode = (mode == "PP")
    allowed_names = {"p", "k"} if pp_mode else {"n"}

    low = expr.lower()
    if ("nan" in low) or ("inf" in low) or ("__" in expr):
        fail("forbidden token")

    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        fail("parse error")
    try:
        validate_ast(tree, allowed_names, pp_mode)
    except ValueError as ex:
        fail("disallowed: %s" % ex)

    n_nodes = sum(1 for _ in ast.walk(tree))

    # ---- withheld split: genuine extrapolation, regenerated deterministically ----
    hold_ns = gen_holdout_ns(test_id)
    a2, a1, a3 = coef_for(test_id)
    true_vals = [f_true(n, a2, a1, a3) for n in hold_ns]

    def predict(n):
        if pp_mode:
            acc = 1
            for (p, k) in factorize(n):
                v = eval_ast(tree, {"p": p, "k": k})
                acc *= v
            return acc
        else:
            return eval_ast(tree, {"n": n})

    errs = []
    for n, tv in zip(hold_ns, true_vals):
        try:
            pred = predict(n)
        except Exception:
            fail("eval error on withheld input")
        try:
            pred = float(pred)
        except Exception:
            fail("non-numeric result")
        if not math.isfinite(pred):
            fail("non-finite result")
        errs.append(clipped_rel_err(pred, tv))

    E = sum(errs) / len(errs)
    pen = 1.0 + COMPLEXITY_RATE * max(0, n_nodes - COMPLEXITY_FREE)
    F = (E + FLOOR) * pen

    # ---- internal baseline B: the fixed trivial predictor f(n) = n ----
    base_errs = [clipped_rel_err(float(n), tv) for n, tv in zip(hold_ns, true_vals)]
    E_base = sum(base_errs) / len(base_errs)
    B = (E_base + FLOOR) * 1.0

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    ratio = max(0.0, min(1.0, ratio))
    print("mode=%s held_out_mre=%.6f baseline_mre=%.6f nodes=%d penalty=%.4f Ratio: %.6f"
          % (mode, E, E_base, n_nodes, pen, ratio))


if __name__ == "__main__":
    main()
