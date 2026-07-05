#!/usr/bin/env python3
# verify.py -- deterministic scorer for the Alloy Constitutive-Law problem (format E).
# CLI:  python3 verify.py <in> <out> <ans>     (ans is ignored)
# Prints exactly one final "Ratio: <float in [0,1]>" line and exits 0.
#
# Anti-leak: the hidden constitutive law + held-out region are regenerated HERE (never shipped
# as an importable module, never printed by gen.py). The held-out split is genuine extrapolation
# (strain and strain rate strictly larger than any training point) so the score rewards
# generalisation, not memorisation.
import sys, math, ast, random

# ---- FIXED grids / hidden law (byte-identical to gen.py) ----
S_TRAIN = [0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
R_TRAIN = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
S_HOLD  = [0.15, 0.22, 0.30, 0.40]
R_HOLD  = [64.0, 128.0, 256.0, 512.0]
NMAX_ID = 12   # search over this many candidate testIds to re-identify the instance

COMPLEXITY_FREE = 40      # AST nodes below this incur no penalty
COMPLEXITY_RATE = 0.01    # per-node penalty above the threshold


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


def law_params(test_id):
    rr = random.Random(4220 + test_id)
    Y = rr.uniform(180.0, 320.0)
    K = rr.uniform(400.0, 900.0)
    n = rr.uniform(0.30, 0.60)
    C = rr.uniform(20.0, 60.0)
    q = rr.uniform(0.08, 0.20)
    return Y, K, n, C, q


def true_val(p, s, r):
    Y, K, n, C, q = p
    return Y + K * (s ** n) + C * (r ** q)


def sigma_for(test_id):
    return 0.020 + 0.004 * test_id


def gen_points(test_id, Ss, Rs, tag):
    p = law_params(test_id)
    sig = sigma_for(test_id)
    rng = random.Random(77000 + test_id * 13 + tag)
    pts = []
    for s in Ss:
        for r in Rs:
            val = true_val(p, s, r) * (1.0 + rng.gauss(0.0, sig))
            pts.append((s, r, val))
    return pts


def rmse(pred, obs):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(pred, obs)) / len(obs))


# ---------- linear least squares (normal equations + Gaussian elimination) ----------
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
    "abs": abs, "pow": pow, "sin": math.sin, "cos": math.cos,
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
    """Raise ValueError on any disallowed construct."""
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
        if node.id not in ("s", "r") and node.id not in _ALLOWED_NAMES:
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


def eval_ast(node, s, r):
    if isinstance(node, ast.Expression):
        return eval_ast(node.body, s, r)
    if isinstance(node, ast.BinOp):
        a = eval_ast(node.left, s, r); b = eval_ast(node.right, s, r)
        op = node.op
        if isinstance(op, ast.Add): return a + b
        if isinstance(op, ast.Sub): return a - b
        if isinstance(op, ast.Mult): return a * b
        if isinstance(op, ast.Div): return a / b
        if isinstance(op, ast.Pow): return a ** b
    if isinstance(node, ast.UnaryOp):
        a = eval_ast(node.operand, s, r)
        return +a if isinstance(node.op, ast.UAdd) else -a
    if isinstance(node, ast.Name):
        if node.id == "s": return s
        if node.id == "r": return r
        return _ALLOWED_NAMES[node.id]
    if isinstance(node, ast.Call):
        args = [eval_ast(a, s, r) for a in node.args]
        return _ALLOWED_FUNCS[node.func.id](*args)
    v = _num(node)
    if v is not None:
        return v
    raise ValueError("eval")


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]

    # ---- read instance (train stress-strain table) ----
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
    if n <= 0 or len(toks) < 1 + 3 * n:
        fail("truncated instance")
    train = []
    idx = 1
    for _ in range(n):
        s = float(toks[idx]); r = float(toks[idx + 1]); d = float(toks[idx + 2])
        idx += 3
        train.append((s, r, d))

    # ---- re-identify which testId produced this instance (fixes the held-out ground truth) ----
    best_id, best_err = None, None
    for tid in range(1, NMAX_ID + 1):
        cand = gen_points(tid, S_TRAIN, R_TRAIN, tag=1)
        if len(cand) != len(train):
            continue
        err = 0.0
        ok = True
        for (s1, r1, d1), (s2, r2, d2) in zip(train, cand):
            if abs(s1 - s2) > 1e-9 or abs(r1 - r2) > 1e-9:
                ok = False
                break
            err = max(err, abs(d1 - d2) / max(1e-9, abs(d2)))
        if not ok:
            continue
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
    expr = raw.strip()
    if not expr:
        fail("empty output")
    for line in raw.splitlines():
        if line.strip():
            expr = line.strip()
            break

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

    # ---- held-out ground truth (extrapolation region), regenerated deterministically ----
    hold = gen_points(test_id, S_HOLD, R_HOLD, tag=2)
    hobs = [d for _, _, d in hold]

    # ---- evaluate submitted expression on held-out region ----
    preds = []
    for (s, r, _d) in hold:
        try:
            val = eval_ast(tree, float(s), float(r))
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

    # complexity penalty (gentle backstop against memorising huge expressions)
    pen = 1.0 + COMPLEXITY_RATE * max(0, n_nodes - COMPLEXITY_FREE)
    F = F_raw * pen

    # ---- internal baseline B: single product power-law via log-linear regression on TRAIN ----
    #   sigma ~ k * s^a * r^b   (ignores the additive yield floor -> mispredicts high strain)
    rows = [[1.0, math.log(s), math.log(r)] for (s, r, _d) in train]
    ylog = [math.log(d) for (_s, _r, d) in train]
    co = lstsq(rows, ylog)
    if co is None:
        fail("internal baseline failed")
    bpred = [math.exp(co[0] + co[1] * math.log(s) + co[2] * math.log(r)) for (s, r, _d) in hold]
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
