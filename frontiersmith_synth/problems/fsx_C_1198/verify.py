#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the queue-saturation-forecast task.

- Reads the test id from the <in> header, then regenerates the SAME hidden
  queue (capacity C, sensitivities alpha/beta, burstiness exponent p) and a
  HELD-OUT set of (load, burstiness) points drawn from a HIGHER offered-load
  range than the training sample -- entirely from that id.  The hidden law
  lives ONLY here (and, necessarily, duplicated verbatim in gen.py -- never
  imported, never printed to the solver).
- Parses the participant's output: ONE closed-form arithmetic expression over
  variables `L` (offered load) and `B` (burstiness), using +,-,*,/,**,
  parentheses and numeric constants only (no function calls, no other names).
- Evaluates that expression at every held-out (L,B) pair.  Any parse error,
  disallowed syntax, non-finite constant/result, or predicted wait < 0
  anywhere scores 0 for the whole test case (feasibility gate).
- Scores held-out MSE (against the held-out sample, which carries its own
  independent measurement noise -- so even the TRUE law does not score a
  perfect 0 MSE) with a mild expression-size parsimony penalty, against an
  internal baseline (checker's own single-parameter proportional fit
  `k*L`, fit on the training rows, ignoring burstiness entirely).  The
  raw MSE ratio is mapped through a bounded saturating curve so a
  baseline-matching submission scores ~0.10 and no finite improvement can
  ever reach the ceiling CAP=0.88 -- there is always genuine headroom above
  even a very good fit (irreducible measurement noise + the deliberately
  unmodelled true burstiness exponent, which only APPROXIMATELY equals 2).
"""
import sys, math, ast, random

CAP = 0.88
BASE_SCORE = 0.10
K = CAP / BASE_SCORE - 1.0     # saturating-curve constant: ratio_raw==1 -> BASE_SCORE
LAMBDA = 0.01                  # parsimony weight
BASE_NODES = 3                 # node count of the baseline's own "k*L" fit
MAX_NODES = 60
MAX_OUT_BYTES = 20000
M_HELDOUT = 25


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden queue (identical derivation to gen.py) ----------
def hidden_params(t):
    rng = random.Random(700003 + t * 91711)
    C = rng.uniform(50.0, 200.0)
    alpha = rng.uniform(0.5, 1.1)
    beta = rng.uniform(0.5, 1.4)
    p = rng.uniform(1.7, 2.3)
    return C, alpha, beta, p


def schedule(t):
    u_train_max = [0.55, 0.52, 0.50, 0.48, 0.46, 0.44, 0.42, 0.40, 0.38, 0.36][t - 1]
    sigma_mult  = [0.02, 0.02, 0.025, 0.025, 0.03, 0.03, 0.035, 0.035, 0.04, 0.04][t - 1]
    n_train     = [70, 70, 65, 65, 60, 60, 55, 55, 50, 50][t - 1]
    return u_train_max, sigma_mult, n_train


def heldout_range(t):
    """Escalating trap: later test ids probe closer to capacity (kept <=0.84 of
    C -- well inside the finite domain -- so this is genuine but bounded
    extrapolation, not a singularity chase)."""
    u_lo = [0.50, 0.52, 0.54, 0.56, 0.58, 0.62, 0.65, 0.68, 0.71, 0.74][t - 1]
    u_hi = [0.68, 0.70, 0.71, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84][t - 1]
    return u_lo, u_hi


def true_wait(L, B, C, alpha, beta, p):
    return (alpha + beta * (B ** p)) * L / (C - L)


def gen_train_rows(t):
    """Reproduce exactly the training rows the solver saw (needed to fit the
    internal baseline -- the baseline is NOT allowed to peek at hidden params,
    only at the same train rows the solver got)."""
    C, alpha, beta, p = hidden_params(t)
    u_train_max, sigma_mult, n_train = schedule(t)
    rng = random.Random(1000003 + t * 7919)
    rows = []
    for _ in range(n_train):
        u = rng.uniform(0.03, u_train_max)
        L = u * C
        B = rng.uniform(0.2, 2.2)
        w = true_wait(L, B, C, alpha, beta, p)
        noise = rng.gauss(0.0, sigma_mult)
        w_obs = w * math.exp(noise)
        rows.append((L, B, w_obs))
    return rows


def gen_heldout_rows(t, C, alpha, beta, p, sigma_mult):
    rng = random.Random(9000007 + t * 15485863)
    u_lo, u_hi = heldout_range(t)
    rows = []
    for _ in range(M_HELDOUT):
        u = rng.uniform(u_lo, u_hi)
        L = u * C
        B = rng.uniform(0.2, 2.2)
        w = true_wait(L, B, C, alpha, beta, p)
        noise = rng.gauss(0.0, sigma_mult)
        rows.append((L, B, w * math.exp(noise)))
    return rows


def baseline_k(train_rows):
    """Checker's own trivial construction: single-parameter proportional fit
    W ~= k*L through the origin (ignores burstiness entirely)."""
    num = sum(L * w for L, B, w in train_rows)
    den = sum(L * L for L, B, w in train_rows)
    return num / den if den > 1e-12 else 0.0


# ---------- expression parsing (arithmetic over L, B only) ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
)


def _count_nodes(tree):
    return sum(1 for nd in ast.walk(tree)
               if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Name, ast.Constant)))


def parse_expression(text):
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        fail("empty output")
    if len(lines) > 1:
        fail("expected a single expression line, got %d non-blank lines" % len(lines))
    expr_text = lines[0].strip()
    if len(expr_text) > 1000:
        fail("expression too long")
    try:
        tree = ast.parse(expr_text, mode="eval")
    except Exception:
        fail("parse error")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            fail("disallowed syntax %s" % type(node).__name__)
        if isinstance(node, ast.Name):
            if node.id not in ("L", "B"):
                fail("unknown name %s (only L, B allowed)" % node.id)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                fail("non-numeric constant")
            v = float(node.value)
            if v != v or v in (float("inf"), float("-inf")):
                fail("non-finite constant")
    nodes = _count_nodes(tree)
    if nodes > MAX_NODES:
        fail("program too large (%d nodes > %d)" % (nodes, MAX_NODES))
    try:
        code = compile(tree, "<expr>", "eval")
    except Exception:
        fail("compile error")
    return code, nodes


def evaluate(code, L, B):
    env = {"L": float(L), "B": float(B)}
    try:
        v = eval(code, {"__builtins__": {}}, env)
    except Exception:
        fail("evaluation error at L=%.4f B=%.4f" % (L, B))
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        fail("non-numeric result")
    v = float(v)
    if v != v or v in (float("inf"), float("-inf")):
        fail("non-finite result at L=%.4f B=%.4f" % (L, B))
    if v < -1e-6:
        fail("negative predicted wait time at L=%.4f B=%.4f (%.6f)" % (L, B, v))
    return max(v, 0.0)


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            header = fh.readline().split()
        t = int(header[1])
    except Exception:
        fail("bad instance header")
    if t < 1 or t > 100000:
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

    C, alpha, beta, p = hidden_params(t)
    u_train_max, sigma_mult, n_train = schedule(t)
    train_rows = gen_train_rows(t)
    heldout = gen_heldout_rows(t, C, alpha, beta, p, sigma_mult)

    se = 0.0
    for L, B, w in heldout:
        pred = evaluate(code, L, B)
        se += (pred - w) ** 2
    F_mse = se / len(heldout)

    k = baseline_k(train_rows)
    base_se = sum((k * L - w) ** 2 for L, B, w in heldout)
    B_mse = base_se / len(heldout)

    F = F_mse * (1.0 + LAMBDA * nodes)
    Bv = B_mse * (1.0 + LAMBDA * BASE_NODES)

    ratio_raw = Bv / max(1e-12, F)
    sc = CAP * ratio_raw / (ratio_raw + K)

    print("heldout_MSE=%.6f baseline_MSE=%.6f nodes=%d  Ratio: %.6f"
          % (F_mse, B_mse, nodes, sc))


if __name__ == "__main__":
    main()
