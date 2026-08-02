#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for traffic-metastable-forecast ("Free flow until one
brake light").

- Reads the test id from <in>, regenerates the SAME hidden road (jam density
  RJ, critical density RC, capacity QM, susceptibility scale S0, metastable
  decline MU, capacity-drop fraction DELTA) and a HELD-OUT set of (rho, P)
  points drawn from a HIGHER density range than training -- straddling RC
  and reaching into the broken-down region -- entirely from the test id.
  The hidden law lives ONLY here (and, necessarily, duplicated verbatim in
  gen.py -- never imported, never printed to the solver).
- Parses the participant's output: ONE closed-form arithmetic expression
  over `rho` (density) and `P` (perturbation magnitude), using +,-,*,/,**,
  parentheses and numeric constants only (no function calls, no other
  names).
- Evaluates that expression at every held-out (rho,P) pair. Any parse
  error, disallowed syntax, non-finite constant/result, or predicted flow
  < 0 anywhere scores 0 for the whole test case (feasibility gate).
- Scores held-out MSE (against held-out flow, which carries its own
  independent measurement noise) with a mild expression-size parsimony
  penalty, against an internal baseline (checker's own trivial
  construction: the CONSTANT mean flow observed on the training rows,
  ignoring density, P and any breakdown behaviour entirely). The raw MSE
  ratio is mapped through a bounded saturating curve so a baseline-matching
  submission scores ~0.10 and no finite improvement can ever reach the
  ceiling CAP=0.88 -- held-out noise plus the fact that the true metastable
  decline MU and capacity-drop DELTA vary a fair amount per road (never
  recoverable from sub-critical-only training data) keep a perfect fit out
  of reach.
"""
import sys, math, ast, random

CAP = 0.88
BASE_SCORE = 0.10
K = CAP / BASE_SCORE - 1.0     # saturating-curve constant: ratio_raw==1 -> BASE_SCORE
LAMBDA = 0.01                  # parsimony weight
BASE_NODES = 1                 # node count of the baseline's own constant fit
MAX_NODES = 150
MAX_OUT_BYTES = 20000
M_HELDOUT = 25
EPS0 = 3.0
PMAX = 25.0


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden road (identical derivation to gen.py) ----------
def hidden_params(t):
    rng = random.Random(812503 + t * 93179)
    RJ = rng.uniform(180.0, 260.0)
    frac_crit = rng.uniform(0.28, 0.38)
    RC = frac_crit * RJ
    QM = rng.uniform(1800.0, 2600.0)
    S0 = rng.uniform(150.0, 400.0)
    MU = rng.uniform(0.15, 0.35)
    DELTA = rng.uniform(0.30, 0.55)
    return RJ, RC, QM, S0, MU, DELTA


def schedule(t):
    u_train_max = [0.90, 0.87, 0.84, 0.81, 0.78, 0.75, 0.72, 0.69, 0.66, 0.63][t - 1]
    sigma_mult  = [0.02, 0.02, 0.025, 0.025, 0.03, 0.03, 0.035, 0.035, 0.04, 0.04][t - 1]
    n_train     = [70, 70, 65, 65, 60, 60, 55, 55, 50, 50][t - 1]
    return u_train_max, sigma_mult, n_train


def heldout_range(t):
    """Escalating trap: later test ids reach farther past RC into the
    broken-down region (kept <= 0.75 of the RC..RJ gap -- genuine but
    bounded extrapolation)."""
    lo_frac = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20][t - 1]
    hi_frac = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.68, 0.72, 0.75][t - 1]
    return lo_frac, hi_frac


def train_flow(rho, P, RC, QM, S0):
    Fval = QM * rho / RC
    chi = S0 / (abs(rho - RC) + EPS0)
    return Fval - chi * P


def gen_train_rows(t):
    """Reproduce exactly the training rows the solver saw (needed to fit the
    internal baseline -- the baseline is NOT allowed to peek at hidden
    params, only at the same train rows the solver got)."""
    RJ, RC, QM, S0, MU, DELTA = hidden_params(t)
    u_train_max, sigma_mult, n_train = schedule(t)
    rng = random.Random(2000011 + t * 7919)
    rows = []
    for _ in range(n_train):
        rho = RC * rng.uniform(0.05, u_train_max)
        P = rng.uniform(0.0, PMAX)
        q_clean = train_flow(rho, P, RC, QM, S0)
        q_clean = max(q_clean, 1.0)
        noise = rng.gauss(0.0, sigma_mult)
        rows.append((rho, P, q_clean * math.exp(noise)))
    return rows


def gen_heldout_rows(t, RJ, RC, QM, S0, MU, DELTA, sigma_mult):
    rng = random.Random(9500009 + t * 15485863)
    lo_frac, hi_frac = heldout_range(t)
    rows = []
    for _ in range(M_HELDOUT):
        frac = rng.uniform(lo_frac, hi_frac)     # x = (rho-RC)/(RJ-RC)
        rho = RC + frac * (RJ - RC)
        P = rng.uniform(0.0, PMAX)
        Fmeta = QM * (1.0 - MU * frac)
        Cval = QM * (1.0 - DELTA) * (1.0 - frac)
        chi = S0 / (abs(rho - RC) + EPS0)
        blend_level = (1.0 - frac) * Fmeta + frac * Cval
        q_clean = blend_level - chi * P
        q_clean = max(q_clean, 1.0)
        noise = rng.gauss(0.0, sigma_mult)
        rows.append((rho, P, q_clean * math.exp(noise)))
    return rows


def baseline_mean(train_rows):
    """Checker's own trivial construction: predict the single CONSTANT mean
    training flow, ignoring density, perturbation and breakdown entirely."""
    return sum(q for rho, P, q in train_rows) / len(train_rows)


# ---------- expression parsing (arithmetic over rho, P only) ----------
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
            if node.id not in ("rho", "P"):
                fail("unknown name %s (only rho, P allowed)" % node.id)
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


def evaluate(code, rho, P):
    env = {"rho": float(rho), "P": float(P)}
    try:
        v = eval(code, {"__builtins__": {}}, env)
    except Exception:
        fail("evaluation error at rho=%.4f P=%.4f" % (rho, P))
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        fail("non-numeric result")
    v = float(v)
    if v != v or v in (float("inf"), float("-inf")):
        fail("non-finite result at rho=%.4f P=%.4f" % (rho, P))
    if v < -1e-6:
        fail("negative predicted flow at rho=%.4f P=%.4f (%.6f)" % (rho, P, v))
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

    RJ, RC, QM, S0, MU, DELTA = hidden_params(t)
    u_train_max, sigma_mult, n_train = schedule(t)
    train_rows = gen_train_rows(t)
    heldout = gen_heldout_rows(t, RJ, RC, QM, S0, MU, DELTA, sigma_mult)

    se = 0.0
    for rho, P, q in heldout:
        pred = evaluate(code, rho, P)
        se += (pred - q) ** 2
    F_mse = se / len(heldout)

    m = baseline_mean(train_rows)
    base_se = sum((m - q) ** 2 for rho, P, q in heldout)
    B_mse = base_se / len(heldout)

    F = F_mse * (1.0 + LAMBDA * nodes)
    Bv = B_mse * (1.0 + LAMBDA * BASE_NODES)

    ratio_raw = Bv / max(1e-12, F)
    sc = CAP * ratio_raw / (ratio_raw + K)

    print("heldout_MSE=%.6f baseline_MSE=%.6f nodes=%d  Ratio: %.6f"
          % (F_mse, B_mse, nodes, sc))


if __name__ == "__main__":
    main()
