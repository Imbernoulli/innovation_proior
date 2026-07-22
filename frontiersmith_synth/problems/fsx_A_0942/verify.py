#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the pressure-vessel leak-forensics task.

- Reads the test id `t` from <in>'s header, then regenerates the hidden leak
  law (c, alpha) EXACTLY as gen.py does (same rng stream, same first two
  draws) -- the hidden law lives ONLY here and in gen.py, never in the
  instance file.
- Regenerates a HELD-OUT set of (L, observed_rate) points in a genuinely
  higher, extrapolated L-regime (never touched by the training telemetry),
  with fixed multiplicative sensor noise -- also derived only from `t`.
- Parses the participant's submitted closed-form expression: a single line
  of arithmetic over the variable `L`, with `+ - * / **`, parentheses and
  numeric literals ONLY (no other names/functions/calls).
- Scores: F = mean squared log-error of the submitted expression against
  the held-out (noisy) observed leak rate.  B = the SAME metric for a
  trivial linear-in-L baseline (alpha forced to 1) fit directly from the
  training telemetry by the checker itself, ignoring the conservation
  correction entirely.  Ratio = min(900, 100*B/F) / 1000.  The 900 cap
  (instead of 1000) keeps a fixed sliver of headroom above ANY submission,
  including the unreachable exact-law oracle, so no reference solution can
  saturate the score.
"""
import sys, math, ast, random

SEED_BASE = 900001
T_DAY = 1.0

C_LO, C_HI = 0.006, 0.020
A_LO, A_HI = 1.2, 1.8

SIGMA_OBS = 0.12
HELD_LO, HELD_HI = 80.0, 300.0
M_HELD = 40

MAX_OUT_BYTES = 4000
MAX_EXPR_CHARS = 300
MAX_NODES = 40
SCORE_CAP = 900.0


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden leak law (identical draw order to gen.py) ----------
def base_rng(t):
    return random.Random(SEED_BASE + t * 7919)


def hidden_params(t):
    rng = base_rng(t)
    c = rng.uniform(C_LO, C_HI)
    alpha = rng.uniform(A_LO, A_HI)
    return c, alpha


def held_out_points(t, c_true, alpha_true):
    rng = random.Random(SEED_BASE + 31337 + t * 104729)
    pts = []
    for _ in range(M_HELD):
        u = rng.random()
        L = HELD_LO * (HELD_HI / HELD_LO) ** u
        true_rate = c_true * (L ** alpha_true)
        obs_rate = true_rate * math.exp(rng.gauss(0.0, SIGMA_OBS))
        pts.append((L, obs_rate))
    return pts


# ---------- instance / baseline ----------
def read_instance(path):
    with open(path) as fh:
        lines = [ln.split() for ln in fh.read().splitlines() if ln.strip()]
    if len(lines) < 2:
        fail("truncated instance")
    try:
        D, t = int(lines[0][0]), int(lines[0][1])
        L0 = float(lines[1][0])
    except Exception:
        fail("bad instance header")
    if t < 1 or t > 1000000 or D < 1 or D > 100000:
        fail("bad instance bounds")
    if len(lines) < 2 + D:
        fail("instance missing day rows")
    rows = []
    for i in range(D):
        try:
            Q, E = float(lines[2 + i][0]), float(lines[2 + i][1])
        except Exception:
            fail("bad day row %d" % i)
        rows.append((Q, E))
    return D, t, L0, rows


def baseline_rate_const(L0, rows):
    """Trivial no-insight baseline: force alpha=1 (linear leak) and fit the
    proportionality constant from the RAW (uncorrected-for-charges) daily
    deltas.  This is exactly what solutions/trivial.py also computes."""
    E_prev = L0
    ks = []
    for Q, E in rows:
        raw = (E_prev - E) / T_DAY
        Lref = (E_prev + E) / 2.0
        if raw > 1e-6 and Lref > 1e-6:
            ks.append(raw / Lref)
        E_prev = E
    if not ks:
        return 1e-4
    k = sum(ks) / len(ks)
    return k if k > 1e-4 else 1e-4


# ---------- expression parsing (variable L only) ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
    ast.USub, ast.UAdd,
)


def parse_expr(raw):
    stripped = raw.strip()
    if not stripped:
        fail("empty output")
    lines = [ln for ln in stripped.splitlines() if ln.strip()]
    if len(lines) != 1:
        fail("output must be exactly one non-empty expression line")
    text = lines[0].strip()
    if len(text) > MAX_EXPR_CHARS:
        fail("expression too long")
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    nodes = 0
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            fail("disallowed syntax %s" % type(node).__name__)
        if isinstance(node, ast.Name):
            if node.id != "L":
                fail("unknown variable %s (only L is allowed)" % node.id)
            nodes += 1
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                fail("non-numeric constant")
            v = float(node.value)
            if not math.isfinite(v):
                fail("non-finite constant")
            nodes += 1
        if isinstance(node, (ast.BinOp, ast.UnaryOp)):
            nodes += 1
    if nodes > MAX_NODES:
        fail("expression too large (%d nodes)" % nodes)
    try:
        code = compile(tree, "<expr>", "eval")
    except Exception:
        fail("compile error")
    return code


def eval_expr(code, L):
    try:
        v = eval(code, {"__builtins__": {}}, {"L": L})
    except Exception:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    if not math.isfinite(v):
        return None
    return v


def score_points(pred_fn, held_pts):
    errs = []
    for L, obs_rate in held_pts:
        pred = pred_fn(L)
        if pred is None:
            return None
        p = pred if pred > 1e-9 else 1e-9
        d = math.log(p) - math.log(obs_rate)
        errs.append(d * d)
    return sum(errs) / len(errs)


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    D, t, L0, rows = read_instance(inf)

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace")

    code = parse_expr(text)

    c_true, alpha_true = hidden_params(t)
    held_pts = held_out_points(t, c_true, alpha_true)

    def pred_fn(L):
        return eval_expr(code, L)

    F = score_points(pred_fn, held_pts)
    if F is None:
        fail("non-finite prediction on a held-out point")

    k = baseline_rate_const(L0, rows)
    B = score_points(lambda L: k * L, held_pts)

    sc = min(SCORE_CAP, 100.0 * B / max(1e-9, F))
    print("heldout_F=%.6f baseline_B=%.6f  Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
