#!/usr/bin/env python3
# verify.py -- deterministic scorer for the Bioreactor Growth-Law Extrapolation (format E).
# CLI:  python3 verify.py <in> <out> <ans>     (ans is ignored)
# Prints exactly one final "Ratio: <float in [0,1]>" line and exits 0.
#
# Anti-leak: the hidden Gompertz law + the LATE (held-out) sampling region are regenerated HERE
# (never shipped as an importable module, never printed by gen.py).  The held-out split is genuine
# EXTRAPOLATION (t = 40..90 h, strictly beyond the t <= 30 h training window) -> rewards a solver
# that recovers the SATURATING family and its plateau, not one that memorises early growth.
#
# Objective = minimise held-out RMSE (with a gentle expression-complexity penalty).  The internal
# baseline B is a crude SATURATING logistic fit (fixed plateau K0 = 1.05*max(train)); because it
# already saturates, a correct Gompertz fit beats it only modestly -> the score has real headroom
# (no auto-cap at 1.0), while a non-saturating exponential fit scores WORSE than the baseline.
import sys, math, ast, random

# ---- FIXED grids / hidden law (byte-identical to gen.py) ----
T_TRAIN = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
T_HOLD  = [40, 46, 52, 58, 64, 72, 80, 90]
NMAX_ID = 14   # search this many candidate testIds to re-identify the instance

COMPLEXITY_FREE = 60      # AST nodes below this incur no penalty
COMPLEXITY_RATE = 0.006   # per-node penalty above the threshold


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


def law_params(test_id):
    r = random.Random(6100 + test_id)
    K = r.uniform(30.0, 90.0)
    b = r.uniform(3.0, 5.0)
    c = r.uniform(0.06, 0.10)
    return K, b, c


def true_val(pp, t):
    K, b, c = pp
    return K * math.exp(-b * math.exp(-c * t))


def sigma_for(test_id):
    return 0.045 + 0.005 * test_id


def gen_points(test_id, ts, tag):
    pp = law_params(test_id)
    sig = sigma_for(test_id)
    rng = random.Random(90000 + test_id * 17 + tag)
    pts = []
    for t in ts:
        n = true_val(pp, t) * (1.0 + rng.gauss(0.0, sig))
        pts.append((t, n))
    return pts


def rmse(pred, obs):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(pred, obs)) / len(obs))


# ---------- ordinary least squares via normal equations + Gaussian elimination ----------
def lstsq(rows, y):
    m = len(rows[0])
    A = [[0.0] * m for _ in range(m)]
    bvec = [0.0] * m
    for r, yy in zip(rows, y):
        for i in range(m):
            bvec[i] += r[i] * yy
            for j in range(m):
                A[i][j] += r[i] * r[j]
    M = [A[i][:] + [bvec[i]] for i in range(m)]
    for c in range(m):
        piv = max(range(c, m), key=lambda rr: abs(M[rr][c]))
        M[c], M[piv] = M[piv], M[c]
        if abs(M[c][c]) < 1e-12:
            return None
        for r in range(m):
            if r != c:
                f = M[r][c] / M[c][c]
                for k in range(c, m + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][m] / M[i][i] for i in range(m)]


# ---------- safe expression evaluator (AST whitelist) ----------
_ALLOWED_NAMES = {"pi": math.pi, "e": math.e}
_ALLOWED_FUNCS = {
    "exp": math.exp, "log": math.log, "log10": math.log10, "sqrt": math.sqrt,
    "abs": abs, "pow": pow, "sin": math.sin, "cos": math.cos, "tanh": math.tanh,
}
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_ALLOWED_UNARY = (ast.UAdd, ast.USub)


def _num(node):
    if isinstance(node, ast.Constant):
        return node.value
    if hasattr(ast, "Num") and isinstance(node, ast.Num):
        return node.n
    return None


def validate_ast(node):
    if isinstance(node, ast.Expression):
        validate_ast(node.body); return
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINOPS):
            raise ValueError("op")
        validate_ast(node.left); validate_ast(node.right); return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARY):
            raise ValueError("unary")
        validate_ast(node.operand); return
    if isinstance(node, ast.Name):
        if node.id != "t" and node.id not in _ALLOWED_NAMES:
            raise ValueError("name %r" % node.id)
        return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise ValueError("call")
        if node.keywords:
            raise ValueError("kw")
        if not (1 <= len(node.args) <= 2):
            raise ValueError("arity")
        for a in node.args:
            validate_ast(a)
        return
    v = _num(node)
    if v is not None:
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ValueError("const")
        return
    raise ValueError("node %s" % type(node).__name__)


def eval_ast(node, t):
    if isinstance(node, ast.Expression):
        return eval_ast(node.body, t)
    if isinstance(node, ast.BinOp):
        a = eval_ast(node.left, t); b = eval_ast(node.right, t)
        op = node.op
        if isinstance(op, ast.Add): return a + b
        if isinstance(op, ast.Sub): return a - b
        if isinstance(op, ast.Mult): return a * b
        if isinstance(op, ast.Div): return a / b
        if isinstance(op, ast.Pow): return a ** b
    if isinstance(node, ast.UnaryOp):
        a = eval_ast(node.operand, t)
        return +a if isinstance(node.op, ast.UAdd) else -a
    if isinstance(node, ast.Name):
        if node.id == "t": return t
        return _ALLOWED_NAMES[node.id]
    if isinstance(node, ast.Call):
        args = [eval_ast(a, t) for a in node.args]
        return _ALLOWED_FUNCS[node.func.id](*args)
    v = _num(node)
    if v is not None:
        return v
    raise ValueError("eval")


def crude_logistic_baseline(train, hold_ts):
    """Internal baseline B: a fixed, un-optimised SATURATING logistic fit.

    Plateau K0 fixed at 1.05*max(train); logit-linear fit of the sigmoid slope/midpoint.
    It saturates (so it does not blow up), but its plateau under-shoots the true stationary
    density -> a beatable-but-nontrivial baseline that leaves headroom for a correct fit.
    """
    nmax = max(n for _, n in train)
    K0 = 1.05 * nmax
    rows, z = [], []
    for (t, n) in train:
        p = n / K0
        p = min(max(p, 1e-6), 1.0 - 1e-6)
        rows.append([1.0, t])
        z.append(math.log(p / (1.0 - p)))
    co = lstsq(rows, z)
    if co is None:
        return None
    a0, a1 = co
    preds = []
    for t in hold_ts:
        s = a0 + a1 * t
        s = min(max(s, -60.0), 60.0)
        preds.append(K0 / (1.0 + math.exp(-s)))
    return preds


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]

    # ---- read instance (early train table) ----
    try:
        with open(in_path) as f:
            toks = f.read().split()
    except Exception:
        fail("no instance")
    if len(toks) < 1:
        fail("empty instance")
    try:
        n = int(toks[0])
    except Exception:
        fail("bad Ntrain")
    if n <= 0 or len(toks) < 1 + 2 * n:
        fail("truncated instance")
    train = []
    idx = 1
    for _ in range(n):
        t = float(toks[idx]); nn = float(toks[idx + 1])
        idx += 2
        train.append((t, nn))

    # ---- re-identify which testId produced this instance (fixes the held-out ground truth) ----
    best_id, best_err = None, None
    for tid in range(1, NMAX_ID + 1):
        cand = gen_points(tid, T_TRAIN, tag=1)
        if len(cand) != len(train):
            continue
        err = 0.0
        for (t1, n1), (t2, n2) in zip(train, cand):
            if t1 != t2:
                err = 1e9
                break
            err = max(err, abs(n1 - n2) / max(1e-9, abs(n2)))
        if best_err is None or err < best_err:
            best_err, best_id = err, tid
    if best_id is None or best_err > 1e-6:
        fail("instance not recognised")
    test_id = best_id

    # ---- read participant expression ----
    try:
        with open(out_path) as f:
            raw = f.read()
    except Exception:
        fail("no output")
    if len(raw) > 5000:
        fail("expression too long")
    expr = ""
    for line in raw.splitlines():
        if line.strip():
            expr = line.strip()
            break
    if not expr:
        fail("empty output")

    low = expr.lower()
    if ("nan" in low) or ("inf" in low) or ("__" in expr):
        fail("forbidden token")

    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        fail("parse error")
    try:
        validate_ast(tree)
    except ValueError as ex:
        fail("disallowed: %s" % ex)

    n_nodes = sum(1 for _ in ast.walk(tree))

    # ---- held-out ground truth (late saturation region), regenerated deterministically ----
    hold = gen_points(test_id, T_HOLD, tag=2)
    hobs = [nv for _, nv in hold]

    # ---- evaluate submitted expression on the held-out region ----
    preds = []
    for (t, _n) in hold:
        try:
            val = eval_ast(tree, float(t))
        except Exception:
            fail("eval error on held-out input")
        try:
            fval = float(val)
        except Exception:
            fail("non-numeric result")
        if not math.isfinite(fval):
            fail("non-finite result")
        preds.append(fval)

    F_raw = rmse(preds, hobs)

    pen = 1.0 + COMPLEXITY_RATE * max(0, n_nodes - COMPLEXITY_FREE)
    F = F_raw * pen

    # ---- internal baseline B: crude saturating logistic fit on the early data ----
    bpred = crude_logistic_baseline(train, T_HOLD)
    if bpred is None:
        fail("internal baseline failed")
    B = rmse(bpred, hobs)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    if ratio < 0.0:
        ratio = 0.0
    if ratio > 1.0:
        ratio = 1.0
    print("held_out_rmse=%.6f baseline_rmse=%.6f nodes=%d penalty=%.4f Ratio: %.6f"
          % (F_raw, B, n_nodes, pen, ratio))


if __name__ == "__main__":
    main()
