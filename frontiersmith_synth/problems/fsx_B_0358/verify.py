#!/usr/bin/env python3
# verify.py -- deterministic scorer for the Rooftop-Garden Water-Demand Scaling Law (format E).
# CLI:  python3 verify.py <in> <out> <ans>     (ans is ignored)
# Prints exactly one final "Ratio: <float in [0,1]>" line and exits 0.
#
# Anti-leak: the hidden law + held-out region are regenerated HERE (never shipped as an
# importable module, never printed by gen.py). The held-out split is genuine extrapolation
# (larger bed area AND hotter canopy than any training point) -> rewards generalisation.
import sys, math, ast, random

# ---- FIXED grids / hidden law (byte-identical to gen.py) ----
A_TRAIN = [6, 8, 10, 13, 17, 22, 28, 36]
H_TRAIN = [4, 6, 9, 13, 18, 25]
A_HOLD  = [55, 75, 100, 140]
H_HOLD  = [38, 55, 80, 120]
NMAX_ID = 12   # search over this many candidate testIds to re-identify the instance

COMPLEXITY_FREE = 40      # AST nodes below this incur no penalty
COMPLEXITY_RATE = 0.01    # per-node penalty above the threshold


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


def law_params(test_id):
    r = random.Random(4200 + test_id)
    D0  = r.uniform(5.0, 20.0)
    C   = r.uniform(0.30, 1.00)
    rho = r.uniform(0.8, 2.0)
    q   = r.uniform(1.20, 1.60)
    return D0, C, rho, q


def true_val(pp, A, H):
    D0, C, rho, q = pp
    return D0 + C * ((A + rho * H) ** q)


def sigma_for(test_id):
    return 0.030 + 0.004 * test_id


def gen_points(test_id, As, Hs, tag):
    pp = law_params(test_id)
    sig = sigma_for(test_id)
    rng = random.Random(77000 + test_id * 13 + tag)
    pts = []
    for a in As:
        for h in Hs:
            d = true_val(pp, a, h) * (1.0 + rng.gauss(0.0, sig))
            pts.append((a, h, d))
    return pts


def rmse(pred, obs):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(pred, obs)) / len(obs))


# ---------- linear algebra (normal equations + Gaussian elimination) ----------
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
        if node.id not in ("A", "H") and node.id not in _ALLOWED_NAMES:
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


def eval_ast(node, A, H):
    if isinstance(node, ast.Expression):
        return eval_ast(node.body, A, H)
    if isinstance(node, ast.BinOp):
        a = eval_ast(node.left, A, H); b = eval_ast(node.right, A, H)
        op = node.op
        if isinstance(op, ast.Add): return a + b
        if isinstance(op, ast.Sub): return a - b
        if isinstance(op, ast.Mult): return a * b
        if isinstance(op, ast.Div): return a / b
        if isinstance(op, ast.Pow): return a ** b
    if isinstance(node, ast.UnaryOp):
        a = eval_ast(node.operand, A, H)
        return +a if isinstance(node.op, ast.UAdd) else -a
    if isinstance(node, ast.Name):
        if node.id == "A": return A
        if node.id == "H": return H
        return _ALLOWED_NAMES[node.id]
    if isinstance(node, ast.Call):
        args = [eval_ast(a, A, H) for a in node.args]
        return _ALLOWED_FUNCS[node.func.id](*args)
    v = _num(node)
    if v is not None:
        return v
    raise ValueError("eval")


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]

    # ---- read instance (train table) ----
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
        a = float(toks[idx]); h = float(toks[idx + 1]); d = float(toks[idx + 2])
        idx += 3
        train.append((a, h, d))

    # ---- re-identify which testId produced this instance (fixes the held-out ground truth) ----
    best_id, best_err = None, None
    for tid in range(1, NMAX_ID + 1):
        cand = gen_points(tid, A_TRAIN, H_TRAIN, tag=1)
        if len(cand) != len(train):
            continue
        err = 0.0
        for (a1, h1, d1), (a2, h2, d2) in zip(train, cand):
            err = max(err, abs(d1 - d2) / max(1e-9, abs(d2)))
            if a1 != a2 or h1 != h2:
                err = 1e9
                break
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
    hold = gen_points(test_id, A_HOLD, H_HOLD, tag=2)
    hobs = [d for _, _, d in hold]

    # ---- evaluate submitted expression on held-out region ----
    preds = []
    for (a, h, _d) in hold:
        try:
            val = eval_ast(tree, float(a), float(h))
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
    rows = [[1.0, math.log(a), math.log(h)] for (a, h, _d) in train]
    ylog = [math.log(d) for (_a, _h, d) in train]
    co = lstsq(rows, ylog)
    if co is None:
        fail("internal baseline failed")
    bpred = [math.exp(co[0] + co[1] * math.log(a) + co[2] * math.log(h)) for (a, h, _d) in hold]
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
