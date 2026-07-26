#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the saturating-kinetics-inference recovery task.

- Reads the test id from <in> (header only), then regenerates BOTH the dilute
  TRAIN assay (to build the checker's own constant-predictor baseline) and the
  HELD-OUT extrapolation assay entirely from that id -- the hidden Vmax, Km,
  catalyst list and RNG seeds live ONLY here and in gen.py (never in the .in
  file, never in the participant's view).
- Parses the participant's single closed-form expression (an algebraic formula
  over variables `S` and `C`: + - * / ** and parentheses, numeric constants
  only -- no function calls, no other names). Rejects disallowed syntax,
  oversized programs, and any non-finite value produced while evaluating it.
- Evaluates the expression on the HELD-OUT points -- which span (a) BOTH the
  in-sample catalyst levels AND catalyst levels never shown in training
  (multi-regime consistency), and (b) substrate concentrations well beyond the
  dilute training envelope (genuine extrapolation into the saturating regime,
  with a deliberately much larger extrapolation gap for a THIRD of the test
  ids -- the "trap" cases).
- Scores from held-out relative error (clipped, so a single wild point cannot
  swamp everything) plus a mild expression-size penalty (minimisation):
      F = clipped_rel_MSE * (1 + LAMBDA * nodes)
      B = clipped_rel_MSE_of_constant(mean(train_rate)) * (1 + LAMBDA * 1)
      Ratio = min(1000, 100*B/F) / 1000
  A constant reproduces the baseline (~0.1). A first-order (linear-in-S)
  recipe fits the dilute training rows almost perfectly but explodes on the
  saturating held-out rows -> stays low, catastrophically so on the trap
  ids. Recovering the true saturating form (and pooling the per-catalyst
  specific rate across regimes) drives the error down -- but held-out
  measurement noise keeps even the best possible fit off the ceiling.
"""
import sys, ast, random, math

LAMBDA = 0.01
CLIP = 3.0
MAX_OUT_BYTES = 20000
MAX_EXPR_CHARS = 300
MAX_NODES = 40
TRAP_IDS = (3, 6, 9)


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden reaction law (identical to gen.py) ----------
def seed_params(test_id):
    rng = random.Random(900001 + test_id * 7919)
    if test_id <= 3:
        n_regimes, n_pts = 2, 6
    elif test_id <= 7:
        n_regimes, n_pts = 3, 8
    else:
        n_regimes, n_pts = 4, 9
    Vmax = rng.uniform(20.0, 70.0)
    Km = rng.uniform(3.0, 14.0)
    noise_frac = rng.uniform(0.03, 0.06)
    noise_frac_ho = rng.uniform(0.16, 0.26)
    pool = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    rng.shuffle(pool)
    C_values = sorted(pool[:n_regimes])
    dilute_frac = rng.uniform(0.28, 0.55)
    return dict(Vmax=Vmax, Km=Km, noise_frac=noise_frac, noise_frac_ho=noise_frac_ho,
                C_values=C_values, n_pts=n_pts, dilute_frac=dilute_frac, pool=pool,
                n_regimes=n_regimes)


def rate_true(S, C, Vmax, Km):
    return Vmax * C * S / (Km + S)


def make_train(test_id, p):
    Vmax, Km = p["Vmax"], p["Km"]
    C_values = p["C_values"]
    n_pts = p["n_pts"]
    dilute_hi = p["dilute_frac"] * Km
    rng = random.Random(31337 + test_id * 13)
    rows = []
    for C in C_values:
        for k in range(n_pts):
            frac = (k + 1) / (n_pts + 1)
            S = 0.04 * Km + frac * (dilute_hi - 0.04 * Km)
            S *= rng.uniform(0.97, 1.03)
            tr = rate_true(S, C, Vmax, Km)
            noisy = tr + rng.gauss(0.0, p["noise_frac"] * tr)
            noisy = max(0.0, noisy)
            rows.append((S, C, noisy))
    return rows


def mult_for(test_id):
    """Multiplier of the TRAIN envelope's max S -- calibrates the extrapolation
    GAP consistently regardless of where the dilute fraction happened to land.
    A third of the ids (the traps) push far deeper into the saturating regime."""
    if test_id in TRAP_IDS:
        return [2.0, 4.0, 8.0, 16.0]
    return [1.1, 1.35, 1.65, 2.0]


def make_heldout(test_id, p):
    Vmax, Km = p["Vmax"], p["Km"]
    C_values = p["C_values"]
    pool = p["pool"]
    new_C = [c for c in pool if c not in C_values][:2]   # unseen catalyst regimes
    mult = mult_for(test_id)
    train_max_S = p["dilute_frac"] * Km
    pts = []
    for C in C_values + new_C:
        for m in mult:
            pts.append((m * train_max_S, C))
    rng2 = random.Random(20260101 + test_id * 97)
    obs = []
    for (S, C) in pts:
        tr = rate_true(S, C, Vmax, Km)
        o = tr + rng2.gauss(0.0, p["noise_frac_ho"] * tr)
        obs.append(max(0.0, o))
    return pts, obs


# ---------- expression parsing / validation ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
)


def parse_expression(raw):
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if not lines:
        fail("empty output")
    text = lines[0].strip()
    if len(text) > MAX_EXPR_CHARS:
        fail("expression too long")
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    nodes = 0
    for node in ast.walk(tree):
        nodes += 1
        if not isinstance(node, _ALLOWED_NODES):
            fail("disallowed syntax %s" % type(node).__name__)
        if isinstance(node, ast.Name):
            if node.id not in ("S", "C"):
                fail("unknown name '%s' (only S, C allowed)" % node.id)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                fail("non-numeric constant")
            v = float(node.value)
            if v != v or v in (float("inf"), float("-inf")):
                fail("non-finite constant")
    if nodes > MAX_NODES:
        fail("expression too large (%d nodes)" % nodes)
    try:
        code = compile(tree, "<expr>", "eval")
    except Exception:
        fail("compile error")
    return code, nodes


def eval_expr(code, S, C):
    try:
        v = eval(code, {"__builtins__": {}}, {"S": S, "C": C})
    except Exception:
        fail("evaluation error at S=%.4f C=%.4f" % (S, C))
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        fail("non-numeric result")
    v = float(v)
    if v != v or v in (float("inf"), float("-inf")):
        fail("non-finite result at S=%.4f C=%.4f" % (S, C))
    return v


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            header = fh.readline().split()
        test_id = int(header[0])
    except Exception:
        fail("bad instance header")
    if test_id < 1 or test_id > 100000:
        fail("bad test id")

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace")

    code, nodes = parse_expression(text)

    p = seed_params(test_id)
    Vmax = p["Vmax"]
    rows = make_train(test_id, p)
    pts, obs = make_heldout(test_id, p)

    se = 0.0
    for (S, C), true_r in zip(pts, obs):
        pred = eval_expr(code, S, C)
        rel = (pred - true_r) / (Vmax * C)
        rel = max(-CLIP, min(CLIP, rel))
        se += rel * rel
    mse_rel = se / len(pts)
    F = mse_rel * (1.0 + LAMBDA * nodes)

    const = sum(r for _, _, r in rows) / len(rows)
    se_b = 0.0
    for (S, C), true_r in zip(pts, obs):
        rel = (const - true_r) / (Vmax * C)
        rel = max(-CLIP, min(CLIP, rel))
        se_b += rel * rel
    mse_b = se_b / len(pts)
    B = mse_b * (1.0 + LAMBDA * 1)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("heldout_MSE=%.6f baseline_MSE=%.6f nodes=%d  Ratio: %.6f"
          % (mse_rel, mse_b, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
