#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the dose-response extrapolation task.

- Reads the test id + Dmax from <in>, then regenerates the hidden dose-response
  law (Hill efficacy + accelerating toxicity) identically to gen.py -- the law
  lives ONLY here (and in gen.py, never imported, never printed).
- Parses the participant's two-line submission:
      EFFICACY <expr>
      TOXICITY <expr>
  Expressions are arithmetic over the variable `d`, the operators
  + - * / ** (and unary -), parentheses, numeric constants, and the unary
  functions exp, log, sqrt, abs. Anything else is rejected (Ratio: 0.0).
- Rolls a fixed decision grid of doses across the FULL allowed range
  [0, Dmax] (almost entirely doses the participant never saw), evaluates the
  participant's own EFFICACY-TOXICITY at every grid dose, and takes d_hat =
  the dose where the participant's OWN curves say net benefit is highest.
  Any non-finite/complex value anywhere on the grid -> Ratio: 0.0.
- Scores by how good d_hat ACTUALLY is, using the true hidden curves:
      frac = (Utrue(d_hat) - Utrue(0)) / (Utrue(d*) - Utrue(0)),  clipped to [0,1]
      Ratio = 0.1 + 0.75 * frac
  where d* is the true optimal dose on the same grid. Recommending "give
  nothing" (d=0) reproduces the floor (~0.1). The 0.75 scale caps the ceiling
  well below 1.0 (parameter-recovery noise keeps even a good fit off d*
  exactly, leaving further headroom on top of that).
"""
import sys, math, ast

MAX_NODES = 120
MAX_OUT_BYTES = 20000
GRID_N = 241


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden dose-response law (identical to gen.py) ----------
def hidden_params(t):
    import random
    rng = random.Random(90173 + t * 104729)
    Emax = rng.uniform(65.0, 95.0)
    n_hill = rng.uniform(1.4, 2.6)
    EC50 = rng.uniform(45.0, 85.0)
    Tbase = rng.uniform(1.5, 4.5)
    q = rng.uniform(1.5, 2.1)
    Dmax = EC50 * (2.2 + 0.10 * (t - 1)) + rng.uniform(-5.0, 5.0)
    ratio_target = 1.3 + 0.07 * (t - 1) + rng.uniform(-0.05, 0.05)
    target_T_at_Dmax = Emax * ratio_target
    Tc = max(1e-6, (target_T_at_Dmax - Tbase) / (Dmax ** q))
    train_frac = 1.05 - 0.03 * (t - 1) + rng.uniform(-0.03, 0.03)
    train_frac = max(0.55, train_frac)
    D_train_max = train_frac * EC50
    sigma_E = 0.8 + 0.05 * (t - 1)
    sigma_T = 0.4 + 0.03 * (t - 1)
    n_train = 16 + t
    return dict(Emax=Emax, n_hill=n_hill, EC50=EC50, Tbase=Tbase, q=q, Tc=Tc,
                Dmax=Dmax, D_train_max=D_train_max, sigma_E=sigma_E, sigma_T=sigma_T,
                n_train=n_train)


def E_true(d, p):
    dn = d ** p['n_hill']
    return p['Emax'] * dn / (p['EC50'] ** p['n_hill'] + dn)


def T_true(d, p):
    return p['Tbase'] + p['Tc'] * (d ** p['q'])


def U_true(d, p):
    return E_true(d, p) - T_true(d, p)


# ---------- restricted expression language ----------
def _safe_exp(x):
    if x > 700.0:
        raise OverflowError("exp overflow")
    return math.exp(x)


def _safe_log(x):
    if x <= 0.0:
        raise ValueError("log domain")
    return math.log(x)


def _safe_sqrt(x):
    if x < 0.0:
        raise ValueError("sqrt domain")
    return math.sqrt(x)


ALLOWED_FUNCS = {"exp": _safe_exp, "log": _safe_log, "sqrt": _safe_sqrt, "abs": abs}

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
)


def validate_and_compile(text):
    text = text.strip()
    if not text:
        fail("empty expression")
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    nodes = 0
    for node in ast.walk(tree):
        nodes += 1
        if not isinstance(node, _ALLOWED_NODES):
            fail("disallowed syntax %s" % type(node).__name__)
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS):
                fail("disallowed call")
            if node.keywords or len(node.args) != 1:
                fail("bad function arity")
        if isinstance(node, ast.Name):
            if node.id != "d" and node.id not in ALLOWED_FUNCS:
                fail("unknown name %s" % node.id)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                fail("non-numeric constant")
            v = float(node.value)
            if not math.isfinite(v):
                fail("non-finite constant")
    if nodes > MAX_NODES:
        fail("expression too large (%d nodes)" % nodes)
    try:
        code = compile(tree, "<expr>", "eval")
    except Exception:
        fail("compile error")
    return code


def eval_expr(code, dval):
    env = dict(ALLOWED_FUNCS)
    env["d"] = dval
    try:
        v = eval(code, {"__builtins__": {}}, env)
    except Exception:
        return None
    if isinstance(v, complex) or not isinstance(v, (int, float)) or isinstance(v, bool):
        return None
    v = float(v)
    if not math.isfinite(v):
        return None
    return v


def parse_output(text):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) != 2:
        fail("expected exactly 2 non-blank lines (EFFICACY ..., TOXICITY ...)")
    eff_code = tox_code = None
    seen = set()
    for ln in lines:
        parts = ln.split(None, 1)
        if len(parts) != 2:
            fail("bad line '%s'" % ln)
        kw = parts[0].upper()
        if kw == "EFFICACY":
            if "E" in seen:
                fail("duplicate EFFICACY")
            seen.add("E")
            eff_code = validate_and_compile(parts[1])
        elif kw == "TOXICITY":
            if "T" in seen:
                fail("duplicate TOXICITY")
            seen.add("T")
            tox_code = validate_and_compile(parts[1])
        else:
            fail("unknown line prefix '%s'" % kw)
    if eff_code is None or tox_code is None:
        fail("missing EFFICACY or TOXICITY line")
    return eff_code, tox_code


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            header = fh.readline().split()
        t = int(header[1])
        Dmax = float(header[2])
    except Exception:
        fail("bad instance header")
    if t < 1 or t > 100000 or not math.isfinite(Dmax) or Dmax <= 0:
        fail("bad test id / Dmax")

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace")

    eff_code, tox_code = parse_output(text)

    p = hidden_params(t)
    grid = [Dmax * i / (GRID_N - 1) for i in range(GRID_N)]

    best = None
    dhat = grid[0]
    for g in grid:
        ev = eval_expr(eff_code, g)
        tv = eval_expr(tox_code, g)
        if ev is None or tv is None:
            fail("non-finite/invalid submitted value at dose %.4f" % g)
        val = ev - tv
        if best is None or val > best:
            best = val
            dhat = g

    Us = [U_true(g, p) for g in grid]
    ustar = max(Us)
    u0 = U_true(0.0, p)
    frac = (U_true(dhat, p) - u0) / max(1e-9, (ustar - u0))
    frac = min(1.0, max(0.0, frac))
    ratio = 0.1 + 0.75 * frac
    print("dhat=%.4f frac=%.4f  Ratio: %.6f" % (dhat, frac, ratio))


if __name__ == "__main__":
    main()
