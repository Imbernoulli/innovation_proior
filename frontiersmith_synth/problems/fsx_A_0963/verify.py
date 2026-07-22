#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for "reconstructing a lost tax code from payroll scrolls".

- Reads the training instance from <in> (testId, N, W_LO, W_HI, dT, then N (wage,hours)
  rows -- exactly what gen.py printed, nothing more).
- Regenerates the hidden schedule (tau_lo, z0, tau_hi) and a HELD-OUT population of
  higher-earning workers ENTIRELY from testId -- the true parameters live only here
  and in gen.py, never in the instance the solver reads.
- Parses the participant's submitted tax-schedule EXPRESSION T(z) (a single line, a
  Python-style arithmetic expression over the variable `z`, +-*/, unary minus,
  min/max/abs calls, and single comparisons z<C / z>=C / ... used as 0/1 indicators).
- Validates T is a sane schedule (T(0)~=0, non-decreasing, implied marginal rate in
  [-0.02, 0.98] everywhere on a bounded probe grid) -- otherwise Ratio: 0.0.
- Simulates each held-out worker's best response to the SUBMITTED schedule with a
  generic (schedule-agnostic) grid-search + refinement optimizer over hours h>=0,
  identical in method to the optimizer used to score the checker's own naive
  flat-rate baseline. Scores mean-squared hours error against noisy held-out ground
  truth, with a small parsimony penalty on expression size.
      F = heldout_MSE * (1 + LAMBDA*nodes)
      B = baseline_MSE  (flat single-rate fit from the training data itself)
      Ratio = min(1000, 100*B/F) / 1000
A flat-rate guess (ignoring the notch) reproduces the baseline (~0.1). A schedule fit
directly to the (wage,hours) CURVE nails the training regime (which the solver did
see) and is uninformed about the unobserved upper bracket. Idiosyncratic held-out
labor-supply noise keeps even a correctly reconstructed schedule below the ceiling.
"""
import sys, math, ast, random
import numpy as np

LAMBDA = 0.0015
MAX_NODES = 60
MAX_EXPR_CHARS = 300
MAX_OUT_BYTES = 4000
G_COARSE = 2000
F_REFINE = 300
REFINE_ROUNDS = 2
H_CAP_MULT = 2.3
HELD_N = 400
HELD_NOISE_SD = 0.12


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden model (identical formulas to gen.py; duplicated on purpose) ----------
def true_params(t):
    rng = random.Random(100003 + t * 7919)
    tau_lo = rng.uniform(0.12, 0.24)
    gap = rng.uniform(0.12, 0.22)
    tau_hi = min(tau_lo + gap, 0.62)
    z0 = rng.uniform(500.0, 1500.0) * (1.0 + 0.10 * (t - 1))
    frac = rng.uniform(0.08, 0.16)
    dZ = frac * z0
    z_star = z0 + dZ
    dT = (1.0 - tau_hi) * dZ * dZ / (2.0 * z_star)
    w0_lo = math.sqrt(z0 / (1.0 - tau_lo))
    w0_hi = math.sqrt(z0 / (1.0 - tau_hi))
    w_star = math.sqrt(z_star / (1.0 - tau_hi))
    W_LO = w0_lo * 0.45
    W_HI = w_star
    N = 2500 + 250 * (t - 1)
    return dict(tau_lo=tau_lo, tau_hi=tau_hi, z0=z0, dT=dT, w0_lo=w0_lo, w0_hi=w0_hi,
                w_star=w_star, W_LO=W_LO, W_HI=W_HI, N=N)


def best_response_true_scalar(w, p):
    tau_lo, tau_hi, z0, dT = p["tau_lo"], p["tau_hi"], p["z0"], p["dT"]
    w0_lo, w0_hi = p["w0_lo"], p["w0_hi"]
    best_u, best_h = -1e18, None
    if w <= w0_lo:
        h = w * (1.0 - tau_lo)
        u = (1.0 - tau_lo) ** 2 * w * w / 2.0
        if u > best_u:
            best_u, best_h = u, h
    h_b = z0 / w
    u_b = (1.0 - tau_lo) * z0 - z0 * z0 / (2.0 * w * w)
    if u_b > best_u:
        best_u, best_h = u_b, h_b
    if w >= w0_hi:
        h_c = w * (1.0 - tau_hi)
        u_c = w * w * (1.0 - tau_hi) ** 2 / 2.0 - dT + z0 * (tau_hi - tau_lo)
        if u_c > best_u:
            best_u, best_h = u_c, h_c
    return best_h


def held_out_sample(t, p):
    rng_w = random.Random(300003 + t * 7919)
    rng_noise = random.Random(400003 + t * 7919)
    lo = p["W_HI"] * 1.02
    hi = lo + 2.0 * (p["W_HI"] - p["W_LO"])
    ws, targets = [], []
    for _ in range(HELD_N):
        w = rng_w.uniform(lo, hi)
        h_true = best_response_true_scalar(w, p)
        h_target = max(h_true * (1.0 + rng_noise.gauss(0.0, HELD_NOISE_SD)), 1e-4)
        ws.append(w)
        targets.append(h_target)
    return np.array(ws), np.array(targets)


# ---------- safe expression parsing (AST whitelist; compiled + eval'd with a locked-down namespace) ----------
_ALLOWED_COMPARE_OPS = (ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq)
_ALLOWED_FUNCS = ("min", "max", "abs")


def _validate_ast(tree):
    n_nodes = 0
    for node in ast.walk(tree):
        n_nodes += 1
        if isinstance(node, ast.Expression):
            continue
        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                return None, "disallowed binary operator"
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.USub, ast.UAdd)):
                return None, "disallowed unary operator"
        elif isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                return None, "chained comparison not allowed"
            if not isinstance(node.ops[0], _ALLOWED_COMPARE_OPS):
                return None, "disallowed comparison operator"
        elif isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS):
                return None, "disallowed function call"
            if node.keywords:
                return None, "keyword args not allowed"
            nargs = 2 if node.func.id in ("min", "max") else 1
            if len(node.args) != nargs:
                return None, "bad function arity"
        elif isinstance(node, ast.Name):
            if node.id != "z" and node.id not in _ALLOWED_FUNCS:
                return None, "unknown identifier %r" % node.id
        elif isinstance(node, ast.Load):
            continue
        elif isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
                                ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq)):
            continue
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                return None, "non-numeric constant"
            v = float(node.value)
            if v != v or v in (float("inf"), float("-inf")):
                return None, "non-finite constant"
        else:
            return None, "disallowed syntax %s" % type(node).__name__
    return n_nodes, None


def parse_schedule(raw):
    text = raw.strip()
    if not text:
        fail("empty submission")
    text = text.splitlines()[0].strip()
    if not text:
        fail("empty submission")
    if len(text) > MAX_EXPR_CHARS:
        fail("expression too long")
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    n_nodes, err = _validate_ast(tree)
    if err:
        fail(err)
    if n_nodes > MAX_NODES:
        fail("expression too large (%d nodes)" % n_nodes)
    try:
        code = compile(tree, "<schedule>", "eval")
    except Exception:
        fail("compile error")
    return code, n_nodes, collect_constants(tree)


def make_T(code):
    ns_funcs = {"min": np.minimum, "max": np.maximum, "abs": np.abs}

    def T(z):
        try:
            with np.errstate(all="ignore"):
                env = dict(ns_funcs); env["z"] = z
                v = eval(code, {"__builtins__": {}}, env)
        except Exception:
            return np.full(np.shape(z), np.nan, dtype=float)
        return np.asarray(v, dtype=float) + np.zeros_like(np.asarray(z, dtype=float))
    return T


def _affine(node):
    """Return (a, b) such that node == a*z + b, if node is an affine function of z
    under this grammar's operators; None if it depends on z non-affinely (e.g. via
    min/max, or z appearing on both sides of a product/ratio -- comparisons built
    from those are handled by the literal-constant fallback below instead)."""
    if isinstance(node, ast.Constant):
        return (0.0, float(node.value))
    if isinstance(node, ast.Name):
        return (1.0, 0.0) if node.id == "z" else None
    if isinstance(node, ast.UnaryOp):
        inner = _affine(node.operand)
        if inner is None:
            return None
        a, b = inner
        return (-a, -b) if isinstance(node.op, ast.USub) else (a, b)
    if isinstance(node, ast.BinOp):
        l, r = _affine(node.left), _affine(node.right)
        if l is None or r is None:
            return None
        al, bl = l; ar, br = r
        if isinstance(node.op, ast.Add):
            return (al + ar, bl + br)
        if isinstance(node.op, ast.Sub):
            return (al - ar, bl - br)
        if isinstance(node.op, ast.Mult):
            if al == 0.0:
                return (bl * ar, bl * br)
            if ar == 0.0:
                return (br * al, br * bl)
            return None  # z * z-dependent -> non-affine
        if isinstance(node.op, ast.Div):
            if ar == 0.0 and br != 0.0:
                return (al / br, bl / br)
            return None  # dividing by a z-dependent value -> non-affine
    return None  # Call (min/max/abs) or anything else: not affine


def _roots_equal(node, C):
    """z-values where node(z) == C, for node built from affine arithmetic possibly
    NESTED inside min/max (a piecewise-affine function): recurse into whichever
    branch(es) could plausibly equal C rather than requiring the whole subtree to be
    a single affine expression. Over-collecting candidates is harmless (just extra
    probe points); the only failure mode we must avoid is under-collecting."""
    aff = _affine(node)
    if aff is not None:
        a, b = aff
        return {(C - b) / a} if a != 0.0 else set()
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in ("min", "max") and len(node.args) == 2:
        return _roots_equal(node.args[0], C) | _roots_equal(node.args[1], C)
    return set()


def collect_constants(tree):
    """Candidate breakpoint locations (in z) for the dense feasibility probe.
    Arithmetic can disguise a threshold's true location (e.g. `z/0.001 - 3500000 >
    0` crosses zero at z=3500, not at the literal 3500000; `max(0, min(1, z/0.001 -
    3500000))` builds the same jump with no Compare node in sight at all), so
    breakpoints are found by symbolically SOLVING, not by pattern-matching literals:
      (a) every Compare node: root(s) of (left - right), recursing through nested
          min/max on either side via `_roots_equal`;
      (b) every min(x,y)/max(x,y) call: where the two arguments cross, likewise
          recursing through further nesting;
      (c) every bare literal constant, as a fallback for anything built even more
          indirectly than (a)/(b) can resolve.
    """
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            diff = ast.BinOp(left=node.left, op=ast.Sub(), right=node.comparators[0])
            roots |= _roots_equal(diff, 0.0)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in ("min", "max") and len(node.args) == 2:
            x, y = node.args
            # Solving where two genuinely nested pieces cross in full generality is
            # circular; instead resolve each branch against whatever constant the
            # OTHER branch reduces to (covers the "affine vs constant, possibly
            # re-nested" constructions this grammar can actually build).
            for a_side, b_side in ((x, y), (y, x)):
                ab = _affine(b_side)
                if ab is not None and ab[0] == 0.0:
                    roots |= _roots_equal(a_side, ab[1])
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            roots.add(float(node.value))
    return sorted(roots)


def check_feasible(T, z_probe_max, constants):
    """A valid tax schedule may have a genuine NOTCH -- an arbitrarily large upward
    jump in T exactly at a declared bracket boundary (the whole point of the family)
    -- but liability must never fall as income rises (that would be a hidden subsidy
    pocket a worker could exploit), and away from any declared threshold the implied
    marginal rate must stay in a sane range. Discontinuities in this grammar can only
    occur at the submission's own literal constants, so:
      (1) monotonicity is checked on a grid that is DENSELY packed around every such
          constant (immune to narrow-window evasion: a decrease can't hide anywhere
          the grammar is capable of placing one);
      (2) the marginal-rate ceiling is checked on the coarse uniform grid only,
          skipping any coarse interval that straddles a declared threshold (where an
          arbitrarily large legitimate upward jump is allowed).
    """
    coarse = np.linspace(0.0, z_probe_max, 240)
    dense = list(coarse)
    offsets = [-1e-1, -1e-3, -1e-5, -1e-7, 0.0, 1e-7, 1e-5, 1e-3, 1e-1]
    for c in constants:
        for off in offsets:
            z = c + off
            if 0.0 <= z <= z_probe_max:
                dense.append(z)
    all_zs = np.array(sorted(set(dense)))
    tv_all = T(all_zs)
    if not np.all(np.isfinite(tv_all)):
        return False, "non-finite T(z)"
    if abs(tv_all[0]) > 1e-4 * max(1.0, z_probe_max) + 1e-4:
        return False, "T(0) far from 0"

    # (1) global monotonicity. Tolerance for float noise only, scaled PER-PAIR from
    # that pair's own two values -- never from the submission's global max, or an
    # unrelated huge jump elsewhere would buy slack for a real decrease right here.
    d = np.diff(tv_all)
    local_scale = np.maximum(np.abs(tv_all[:-1]), np.abs(tv_all[1:]))
    tol = 1e-6 * np.maximum(1.0, local_scale)
    if np.min(d + tol) < 0.0:
        return False, "tax liability decreases somewhere (hidden subsidy pocket)"

    # (2) marginal-rate ceiling, coarse grid, skipping intervals with a declared jump
    tv_coarse = T(coarse)
    thr = np.array(constants, dtype=float) if constants else np.array([])
    for i in range(len(coarse) - 1):
        z1, z2 = coarse[i], coarse[i + 1]
        if thr.size and np.any((thr >= z1) & (thr <= z2)):
            continue
        slope = (tv_coarse[i + 1] - tv_coarse[i]) / (z2 - z1)
        if slope > 0.98:
            return False, "implausible marginal rate"
    return True, None


# ---------- generic best-response solver (schedule-agnostic, vectorized grid search) ----------
def solve_hours_batch(w, T):
    M = w.shape[0]
    hi = H_CAP_MULT * w
    lo = np.zeros(M)
    t = np.linspace(0.0, 1.0, G_COARSE)
    h_grid = lo[:, None] + t[None, :] * (hi - lo)[:, None]
    z_grid = w[:, None] * h_grid
    Tv = T(z_grid)
    Tv = np.where(np.isfinite(Tv), Tv, np.inf)
    U = w[:, None] * h_grid - Tv - h_grid ** 2 / 2.0
    idx = np.argmax(U, axis=1)
    best_h = h_grid[np.arange(M), idx]
    span = (hi - lo) / (G_COARSE - 1)
    for _ in range(REFINE_ROUNDS):
        rlo = np.maximum(0.0, best_h - span)
        rhi = best_h + span
        tt = np.linspace(0.0, 1.0, F_REFINE)
        hg = rlo[:, None] + tt[None, :] * (rhi - rlo)[:, None]
        zg = w[:, None] * hg
        Tv2 = T(zg)
        Tv2 = np.where(np.isfinite(Tv2), Tv2, np.inf)
        U2 = w[:, None] * hg - Tv2 - hg ** 2 / 2.0
        idx2 = np.argmax(U2, axis=1)
        best_h = hg[np.arange(M), idx2]
        span = (rhi - rlo) / (F_REFINE - 1)
    return best_h


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            lines = fh.read().split("\n")
        header = lines[0].split()
        t = int(header[0]); N = int(header[1])
        train_w, train_h = [], []
        for i in range(N):
            parts = lines[1 + i].split()
            train_w.append(float(parts[0])); train_h.append(float(parts[1]))
    except Exception:
        fail("bad instance file")
    if t < 1 or t > 100000 or N != len(train_w):
        fail("bad instance header")

    p = true_params(t)

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace")

    code, n_nodes, constants = parse_schedule(text)
    T = make_T(code)

    held_w, held_target = held_out_sample(t, p)
    # must cover the FULL domain the best-response solver can later explore
    # (h up to H_CAP_MULT*w, so z up to H_CAP_MULT*w^2), plus a safety margin --
    # feasibility must not stop short of what solve_hours_batch actually probes.
    z_probe_max = 1.1 * H_CAP_MULT * float(np.max(held_w)) ** 2

    ok, reason = check_feasible(T, z_probe_max, constants)
    if not ok:
        fail("infeasible schedule: %s" % reason)

    preds = solve_hours_batch(held_w, T)
    if not np.all(np.isfinite(preds)) or np.any(preds < 0):
        fail("non-finite/negative predicted hours")

    F_mse = float(np.mean((preds - held_target) ** 2))
    F = F_mse * (1.0 + LAMBDA * n_nodes)

    # baseline: a flat rate fitted from only the LOWER HALF of the archived wage
    # range (the naive analyst never notices anything happens near the top of the
    # window, let alone what lies beyond it)
    order = sorted(range(len(train_w)), key=lambda i: train_w[i])
    half = max(len(order) // 2, 1)
    bot_idx = order[:half]
    sum_w = sum(train_w[i] for i in bot_idx)
    sum_h = sum(train_h[i] for i in bot_idx)
    tau_bar = 1.0 - sum_h / sum_w if sum_w > 0 else 0.0
    tau_bar = min(max(tau_bar, 0.0), 0.95)
    base_preds = np.maximum(held_w * (1.0 - tau_bar), 0.0)
    B_mse = float(np.mean((base_preds - held_target) ** 2))
    B = max(B_mse, 1e-9)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("heldout_MSE=%.6f baseline_MSE=%.6f nodes=%d tau_bar=%.4f  Ratio: %.6f"
          % (F_mse, B_mse, n_nodes, tau_bar, sc / 1000.0))


if __name__ == "__main__":
    main()
