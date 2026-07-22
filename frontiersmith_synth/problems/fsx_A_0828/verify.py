#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for integer-recurrence-recovery / "far-term extrapolation".

- Reads the header "<T> <t>" from <in>, regenerates the hidden law (order D,
  integer roots, integer coefficients) entirely from t (identical code to
  gen.py; the law lives ONLY here + gen.py, never printed to the solver).
- Regenerates a set of HELD-OUT extrapolation indices, all well past the
  visible window T, and computes the true (exact, big-int) sequence value at
  each -- this is genuine extrapolation, never shown to the solver.
- Parses the participant's single-line arithmetic EXPRESSION in variable `n`
  (whitelisted AST: + - * / ** , parentheses, numeric constants, `n`; no
  calls, no other names) and evaluates it at each held-out index.
- Scores from held-out relative error plus a small parsimony (node-count)
  penalty, minimisation, normalised against the checker's own two-point
  geometric-ratio extrapolation baseline.
"""
import sys, ast

LAMBDA = 0.008
MAX_OUT_BYTES = 20000
MAX_NODES = 140
MAX_CONST_EXP = 400
MAX_CONST_BASE = 10 ** 6


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden law (IDENTICAL to gen.py) ----------
def num_trap(t):
    if t <= 2:
        return 2
    if t <= 4:
        return 3
    if t <= 6:
        return 4
    return 5


_C_BASE = {3: 150, 4: 280, 5: 1400, 6: 13000}


def law(t):
    R = 9
    d = num_trap(t)
    D = 1 + d
    T = 2 * D + 6
    C = _C_BASE[D] + 8 * t
    roots = [R] + [R + k for k in range(1, d + 1)]
    coeffs = [C] + [1 if k % 2 == 1 else -1 for k in range(1, d + 1)]
    return roots, coeffs, D, T


def a_of(n, roots, coeffs):
    return sum(c * (r ** n) for c, r in zip(coeffs, roots))


HELDOUT_OFFSETS = [3, 6, 10, 15, 22, 32, 45, 62]


# ---------- expression parsing / validation ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
    ast.USub, ast.UAdd,
)


def _is_pure_const(node):
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _is_pure_const(node.operand)
    return False


def _const_val(node):
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.UnaryOp):
        v = _const_val(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    raise ValueError


def validate_expr(text):
    text = text.strip()
    if not text:
        fail("empty expression")
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    n_nodes = 0
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            fail("disallowed syntax %s" % type(node).__name__)
        if isinstance(node, ast.Name):
            if node.id != "n":
                fail("unknown name %r (only 'n' allowed)" % node.id)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                fail("non-numeric constant")
            v = float(node.value)
            if v != v or v in (float("inf"), float("-inf")):
                fail("non-finite constant")
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Name, ast.Constant)):
            n_nodes += 1
    if n_nodes > MAX_NODES:
        fail("expression too large (%d nodes)" % n_nodes)
    # guard against DoS via a huge constant exponent / constant base
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if _is_pure_const(node.right) and abs(_const_val(node.right)) > MAX_CONST_EXP:
                fail("constant exponent too large")
            if _is_pure_const(node.left) and abs(_const_val(node.left)) > MAX_CONST_BASE:
                fail("constant base too large")
    try:
        code = compile(tree, "<expr>", "eval")
    except Exception:
        fail("compile error")
    return code, n_nodes


def eval_at(code, n_val):
    try:
        v = eval(code, {"__builtins__": {}}, {"n": n_val})
    except Exception:
        fail("evaluation error at n=%d" % n_val)
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        fail("non-numeric result at n=%d" % n_val)
    v = float(v)
    if v != v or v in (float("inf"), float("-inf")):
        fail("non-finite result at n=%d" % n_val)
    return v


def rel_err(pred, true):
    return min(1.0, abs(pred - float(true)) / max(1.0, abs(float(true))))


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            lines = fh.read().splitlines()
        T_hdr, t = map(int, lines[0].split())
    except Exception:
        fail("bad instance header")
    if t < 1 or t > 100000:
        fail("bad test id")

    roots, coeffs, D, T = law(t)
    if T_hdr != T:
        fail("instance/header mismatch")

    # sanity: reparse visible rows (not strictly needed for scoring, but
    # confirms the instance file is well-formed)
    try:
        vis = []
        for ln in lines[1:1 + T]:
            ni, ai = ln.split()
            vis.append((int(ni), int(ai)))
        if len(vis) != T or [p[0] for p in vis] != list(range(T)):
            fail("malformed instance rows")
        last1 = vis[T - 1][1]
        last2 = vis[T - 2][1]
    except Exception:
        fail("cannot parse instance rows")

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace").strip()
    # only the FIRST non-empty line is the expression
    first_line = next((ln for ln in text.splitlines() if ln.strip()), "")
    code, nodes = validate_expr(first_line)

    H = [T + o for o in HELDOUT_OFFSETS]
    errs = []
    for n_val in H:
        true_val = a_of(n_val, roots, coeffs)
        pred = eval_at(code, n_val)
        errs.append(rel_err(pred, true_val))
    mean_err = sum(errs) / len(errs)
    F = mean_err + LAMBDA * nodes

    # checker's own trivial baseline: single geometric term extrapolated from
    # the last two visible points (exact rational arithmetic)
    from fractions import Fraction as Fr
    ratio = Fr(last1, last2)
    base_errs = []
    for n_val in H:
        true_val = a_of(n_val, roots, coeffs)
        bpred = Fr(last1) * (ratio ** (n_val - (T - 1)))
        base_errs.append(min(1.0, abs(float(bpred) - float(true_val)) / max(1.0, abs(float(true_val)))))
    base_mean_err = sum(base_errs) / len(base_errs)
    B = base_mean_err + LAMBDA * 5

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("mean_heldout_relerr=%.6f baseline_relerr=%.6f nodes=%d D=%d  Ratio: %.6f"
          % (mean_err, base_mean_err, nodes, D, sc / 1000.0))


if __name__ == "__main__":
    main()
