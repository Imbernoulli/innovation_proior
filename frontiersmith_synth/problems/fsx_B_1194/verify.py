#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the wake-shedding onset forecast task (a supercritical
Hopf bifurcation).  The solver submits ONE closed-form expression predicting
the oscillation AMPLITUDE as a function of the flow parameter `R`.

- Reads the test id from <in> (header), then regenerates the hidden bifurcation
  law (Rc, a, L) and the HELD-OUT grid entirely from that id.  The held-out grid
  deliberately spans BOTH subcritical R (true amplitude 0) and, crucially,
  SUPERCRITICAL R the solver never saw a single training row for -- some just
  past onset, some far past it.  The law and its coefficients live ONLY here.
- Parses the submitted expression with a strict AST whitelist:
      name      R
      operators + - * / **  and unary +/-
      functions sqrt log exp sig tanh relu absv
      numeric constants
- Evaluates it on the held-out grid, computes a mean-squared error against the
  (noisy) true amplitude, adds a small node-count parsimony penalty, and scores
  against the internal baseline "always predict zero shedding" (minimisation):
      F = MSE * (1 + LAMBDA * nodes)
      B = MSE_of_constant_0 * (1 + LAMBDA * 1)
      Ratio = min(1000, 100*B/F) / 1000
  A constant-zero predictor (or any curve fit ONLY to the training amplitude
  column, which is noise around zero end to end) reproduces the baseline
  exactly (~0.1) and can never beat it -- no training row ever showed
  shedding. Extrapolating the DECAY RATE's linear approach to zero recovers
  the growth-rate law g(R)=a(R-Rc) and beats the baseline, but reporting that
  growth rate itself AS the amplitude ignores the cubic (Landau) saturation
  that the given constant L encodes, and systematically mis-sizes the
  oscillation. Only A(R)=sqrt(relu(g(R))/L) tracks the true post-onset branch;
  held-out sensor noise plus finite-sample fit error keep even that well below
  the ceiling, leaving headroom.
"""
import sys, math, ast, random

# ---- fixed design constants (mirrored byte-for-byte in gen.py) ----
RC_LO, RC_HI = 40.0, 140.0
A_LO, A_HI = 0.010, 0.022
L_LO, L_HI = 0.06, 0.15

# ---- held-out / scoring constants (grader only) ----
N_HELD_EACH = 12          # 4 bands x 12 points = 48 held-out points
NOISE_HELD_ABS = 0.45
NOISE_HELD_REL = 0.28
LAMBDA = 0.01
MAX_NODES = 60
MAX_OUT_BYTES = 100000

ALLOWED_FUNCS = {
    "sqrt": lambda x: math.sqrt(x),
    "log":  lambda x: math.log(x),
    "exp":  lambda x: math.exp(max(-700.0, min(700.0, x))),
    "sig":  lambda x: 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x)))),
    "tanh": math.tanh,
    "relu": lambda x: x if x > 0 else 0.0,
    "absv": abs,
}
ALLOWED_NAMES = {"R"}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden bifurcation law (identical to gen.py) ----------
def params(t):
    rng = random.Random(8100000 + t * 3141593)
    Rc = rng.uniform(RC_LO, RC_HI)
    a = rng.uniform(A_LO, A_HI)
    L = rng.uniform(L_LO, L_HI)
    return Rc, a, L


def true_amp(R, Rc, a, L):
    g = a * (R - Rc)
    if g <= 0.0:
        return 0.0
    return math.sqrt(g / L)


def gen_held(t):
    """Held-out grid: subcritical + near-onset + moderate + far-supercritical bands,
    regenerated ONLY here (never printed by gen.py)."""
    Rc, a, L = params(t)
    rng = random.Random(424242 + t * 101)
    pts = []
    bands = [(-140.0, -5.0), (2.0, 20.0), (25.0, 90.0), (120.0, 190.0)]
    for lo, hi in bands:
        for _ in range(N_HELD_EACH):
            off = rng.uniform(lo, hi)
            R = Rc + off
            A = true_amp(R, Rc, a, L)
            sigma = NOISE_HELD_ABS + NOISE_HELD_REL * A
            Ameas = max(0.0, A + rng.gauss(0.0, sigma))
            pts.append((R, Ameas))
    return pts


# ---------- expression parsing / validation ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
    ast.USub, ast.UAdd,
)


def _validate(tree):
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return "disallowed syntax %s" % type(node).__name__
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS):
                return "disallowed call"
            if node.keywords or len(node.args) != 1:
                return "bad function arity"
        if isinstance(node, ast.Name):
            if node.id in ALLOWED_FUNCS or node.id in ALLOWED_NAMES:
                continue
            return "unknown name %s" % node.id
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                return "non-numeric constant"
            v = float(node.value)
            if v != v or v in (float("inf"), float("-inf")):
                return "non-finite constant"
    return None


def _count_nodes(tree):
    return sum(1 for nd in ast.walk(tree)
               if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant)))


def parse_expr(raw):
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        fail("empty submission")
    text = lines[-1]
    if text.upper().startswith("EXPR "):
        text = text[5:].strip()
    if not text:
        fail("empty expression")
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    err = _validate(tree)
    if err:
        fail(err)
    nodes = _count_nodes(tree)
    if nodes > MAX_NODES:
        fail("expression too large (%d nodes)" % nodes)
    try:
        code = compile(tree, "<expr>", "eval")
    except Exception:
        fail("compile error")
    return code, nodes


def eval_at(code, R):
    env = dict(ALLOWED_FUNCS)
    env["R"] = R
    try:
        p = eval(code, {"__builtins__": {}}, env)
    except Exception:
        return None
    if isinstance(p, bool) or not isinstance(p, (int, float)):
        return None
    p = float(p)
    if p != p or p in (float("inf"), float("-inf")):
        return None
    return p


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

    code, nodes = parse_expr(text)

    held = gen_held(t)
    se = 0.0
    bese = 0.0
    for R, Ameas in held:
        p = eval_at(code, R)
        if p is None:
            fail("non-finite / invalid prediction")
        se += (p - Ameas) ** 2
        bese += Ameas ** 2
    F_mse = se / len(held)
    B_mse = bese / len(held)

    B = B_mse * (1.0 + LAMBDA * 1)
    F = F_mse * (1.0 + LAMBDA * nodes)
    sc = min(1000.0, 100.0 * B / max(1e-12, F))
    print("heldout_MSE=%.6f baseline_MSE=%.6f nodes=%d  Ratio: %.6f"
          % (F_mse, B_mse, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
