#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for "Highway sensor loops before the jam".

- Reads the test id from <in>'s header, then regenerates the hidden flux
  f_true(rho) = V*rho*(1-rho)*(1+S*(rho-0.5)) and a STEEP, high-density,
  held-out initial condition entirely from that id.  The hidden law lives
  ONLY here (and, identically, inside gen.py -- never printed to the
  participant, never in an importable module).
- Parses the participant's output as ONE Python expression over the variable
  `rho`, using + - * / ** and the unary functions
      exp log sin cos sqrt tanh abs
  Any other name/call, non-finite constant, or evaluation failure -> Ratio 0.
- Feeds the expression, as a flux function, into a FIXED Lax-Friedrichs
  entropy-respecting scheme (identical code to gen.py, clipped to [0,1] each
  step) and evolves the held-out steep initial condition forward.  The same
  scheme also evolves (a) the TRUE flux and (b) a zero-flux baseline from the
  same initial condition, for the same number of steps.
- Score = mean-absolute-error of the candidate's simulated profile vs the
  true profile, at several checkpoint times, with a small parsimony penalty
  on the expression's AST node count, normalised against the zero-flux
  baseline's own error (minimisation):
      F = MAE_candidate * (1 + LAMBDA * nodes)
      B = MAE_zero_flux  * (1 + LAMBDA * 1)
      Ratio = min(1000, 100*B/F) / 1000
  Submitting `0` reproduces B exactly (Ratio = 0.1).  A flux that only
  mimics the smooth training profiles (e.g. a single translation speed)
  cannot predict where/how fast the held-out jam's shockwave forms, so it
  stays far from the true evolution; a flux recovered via the CONSERVED
  quantity's own characteristics generalises to the shock for free.
"""
import sys, ast, math, random
import numpy as np

np.seterr(all="ignore")

LAMBDA = 0.004
MAX_NODES = 40
MAX_OUT_BYTES = 20000

N_GRID = 240
T_HOLD = 0.35
CFL = 0.45
SNAP_FRACS = (0.35, 0.65, 1.0)

ALLOWED_FUNCS = {
    "exp": np.exp, "log": np.log, "sin": np.sin, "cos": np.cos,
    "sqrt": np.sqrt, "tanh": np.tanh, "abs": np.abs,
}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------------- hidden law (identical to gen.py) ----------------
def flux_params(t):
    rng = random.Random(9173231 + t * 104729)
    V = rng.uniform(0.85, 1.35)
    S = rng.uniform(-0.55, 0.55)
    W = rng.uniform(0.05, 0.12)
    freq = rng.choice([2, 3])
    return V, S, W, freq


def f_true_vec(rho, V, S, W, freq):
    base = V * rho * (1.0 - rho) * (1.0 + S * (rho - 0.5))
    wiggle = W * V * rho * (1.0 - rho) * np.sin(2.0 * np.pi * freq * rho)
    return base + wiggle


def fprime_true_vec(rho, V, S, W, freq):
    eps = 1e-4
    rp = np.clip(rho + eps, 0.0, 1.0)
    rm = np.clip(rho - eps, 0.0, 1.0)
    return (f_true_vec(rp, V, S, W, freq) - f_true_vec(rm, V, S, W, freq)) / (rp - rm)


def periodic_dist(x, c):
    d = x - c
    d = d - np.round(d)
    return d


def bump_vec(x, c, w, h):
    d = periodic_dist(x, c)
    out = np.zeros_like(x)
    mask = np.abs(d) < w
    out[mask] = h * 0.5 * (1.0 + np.cos(np.pi * d[mask] / w))
    return out


def heldout_ic_params(t):
    rng = random.Random(881221 + t * 104743)
    frac = (t - 1) / 9.0
    rho_L = rng.uniform(0.06, 0.16)
    rho_R = rng.uniform(0.55, 0.65) + 0.20 * frac
    eps = rng.uniform(0.045, 0.06) - 0.03 * frac
    eps = max(0.015, eps)
    return rho_L, rho_R, eps


def heldout_ic(x, t):
    rho_L, rho_R, eps = heldout_ic_params(t)
    h = rho_R - rho_L
    rho = np.full_like(x, rho_L) + bump_vec(x, 0.5, eps, h)
    return np.clip(rho, 0.0, 1.0)


# ---------------- fixed entropy-respecting solver (identical to gen.py) ----------------
def lxf_evolve_snaps(rho0, flux_fn, dx, dt, step_marks):
    """Evolve up to max(step_marks) total steps; return dict {step: profile}.
    Returns None if any non-finite value is ever produced."""
    rho = rho0.copy()
    out = {}
    total = max(step_marks)
    marks = set(step_marks)
    if 0 in marks:
        out[0] = rho.copy()
    for s in range(1, total + 1):
        fr = flux_fn(rho)
        if not np.all(np.isfinite(fr)):
            return None
        rho_l = np.roll(rho, 1); rho_r = np.roll(rho, -1)
        f_l = np.roll(fr, 1); f_r = np.roll(fr, -1)
        rho = 0.5 * (rho_l + rho_r) - (dt / (2.0 * dx)) * (f_r - f_l)
        if not np.all(np.isfinite(rho)):
            return None
        rho = np.clip(rho, 0.0, 1.0)
        if s in marks:
            out[s] = rho.copy()
    return out


# ---------------- expression parsing / validation ----------------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
)


def _count_nodes(tree):
    return sum(1 for nd in ast.walk(tree)
               if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant)))


def parse_expr(text):
    text = text.strip()
    if not text:
        fail("empty output")
    if len(text.splitlines()) > 1:
        # allow trailing blank lines only; otherwise reject multi-statement input
        nonblank = [ln for ln in text.splitlines() if ln.strip()]
        if len(nonblank) != 1:
            fail("output must be a single expression line")
        text = nonblank[0]
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            fail("disallowed syntax %s" % type(node).__name__)
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS):
                fail("disallowed call")
            if node.keywords or len(node.args) != 1:
                fail("bad function arity")
        if isinstance(node, ast.Name):
            if node.id in ALLOWED_FUNCS:
                continue
            if node.id != "rho":
                fail("unknown name %s" % node.id)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                fail("non-numeric constant")
            v = float(node.value)
            if v != v or v in (float("inf"), float("-inf")):
                fail("non-finite constant")
    nodes = _count_nodes(tree)
    if nodes > MAX_NODES:
        fail("expression too large (%d nodes)" % nodes)
    try:
        code = compile(tree, "<expr>", "eval")
    except Exception:
        fail("compile error")
    return code, nodes


def make_flux_fn(code):
    glob = {"__builtins__": {}}
    def fn(rho_arr):
        env = dict(ALLOWED_FUNCS)
        env["rho"] = rho_arr
        try:
            val = eval(code, glob, env)
        except Exception:
            return None
        val = np.asarray(val, dtype=np.float64)
        if val.shape != rho_arr.shape:
            val = np.broadcast_to(val, rho_arr.shape).copy()
        return val
    return fn


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            header = fh.readline().split()
        t = int(header[2])
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
    cand_fn = make_flux_fn(code)

    # cheap pre-check: candidate must be finite on the whole density domain
    probe = np.linspace(0.0, 1.0, 101)
    pv = cand_fn(probe)
    if pv is None or not np.all(np.isfinite(pv)):
        fail("expression not finite/evaluable on rho in [0,1]")

    # ---- regenerate hidden law + held-out steep initial condition ----
    V, S, W, freq = flux_params(t)
    x = np.linspace(0.0, 1.0, N_GRID, endpoint=False)
    dx = 1.0 / N_GRID
    rho0 = heldout_ic(x, t)

    rr = np.linspace(0.0, 1.0, 401)
    max_speed = float(np.max(np.abs(fprime_true_vec(rr, V, S, W, freq))))
    dt = CFL * dx / max(max_speed, 0.3)
    steps_total = max(3, int(T_HOLD / dt))
    marks = sorted(set(max(1, int(round(fr * steps_total))) for fr in SNAP_FRACS))

    def flux_true(r):
        return f_true_vec(r, V, S, W, freq)

    def flux_zero(r):
        return np.zeros_like(r)

    def flux_cand(r):
        v = cand_fn(r)
        if v is None:
            return np.full_like(r, np.nan)
        return v

    true_snaps = lxf_evolve_snaps(rho0, flux_true, dx, dt, marks)
    if true_snaps is None:
        fail("internal error: true evolution unstable")
    base_snaps = lxf_evolve_snaps(rho0, flux_zero, dx, dt, marks)
    cand_snaps = lxf_evolve_snaps(rho0, flux_cand, dx, dt, marks)
    if cand_snaps is None:
        fail("candidate flux produced non-finite values during evolution")

    def mean_mae(snapsA, snapsB):
        errs = [float(np.mean(np.abs(snapsA[s] - snapsB[s]))) for s in marks]
        return sum(errs) / len(errs)

    cand_mae = mean_mae(cand_snaps, true_snaps)
    base_mae = mean_mae(base_snaps, true_snaps)

    F = cand_mae * (1.0 + LAMBDA * nodes)
    B = base_mae * (1.0 + LAMBDA * 1)
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("cand_MAE=%.6f base_MAE=%.6f nodes=%d steps=%d  Ratio: %.6f"
          % (cand_mae, base_mae, nodes, steps_total, sc / 1000.0))


if __name__ == "__main__":
    main()
