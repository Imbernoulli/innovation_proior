#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for corrosion-rate-extrapolate (fsx_C_1330).

- Reads the test id from <in>'s header, then regenerates (a) the hidden
  chemistry/kinetics EXACTLY as gen.py does (duplicated here, never imported,
  never printed) and (b) a HELD-OUT set of environments the solver never saw:
  a mix of near-threshold, clearly-active, and outright "chemistry never
  tested" (new T/pH range entirely, always past the chloride threshold)
  points -- genuine extrapolation, regenerated fresh here only.
- Parses the participant's single-line closed-form expression for R(Cl,T,pH,
  tex,D) via a whitelisted AST (numeric literals; +,-,*,/,**; unary minus;
  variables Cl,T,pH,tex,D; functions exp,log,sqrt,max,min,abs). Any disallowed
  syntax/name, any non-finite or non-positive prediction anywhere in the
  held-out set -> Ratio 0.0 (physical corrosion rates must be finite and
  positive).
- Score: mean absolute log10 error (MAE) against the held-out rates, compared
  against the checker's own baseline (a constant predictor = geometric mean
  of the SAME training rates the solver saw):
      Ratio = min(850, 100 * MAE_baseline / max(1e-9, MAE)) / 1000
  A constant/no-signal predictor reproduces the baseline (Ratio ~ 0.1). The
  850 soft cap (0.85 max) keeps headroom above even a solver that recovers
  the breakdown term well, since the exact jump-rate/curvature is never
  revealed by the (all-passive) training data.
"""
import sys, math, ast

MAX_OUT_BYTES = 20000
MAX_NODES = 150
N_HELD = 30

ALLOWED_FUNCS_ARITY = {
    "exp": 1, "log": 1, "sqrt": 1, "abs": 1, "max": 2, "min": 2,
}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden chemistry/kinetics (identical to gen.py) ----------
def hidden_params(t):
    import random
    rng = random.Random(311007 + t * 7919)
    return dict(
        A0=rng.uniform(0.0008, 0.0022),
        EaR=rng.uniform(3400.0, 5200.0),
        n_t=rng.uniform(-0.28, -0.08),
        b_cl=rng.uniform(0.012, 0.028),
        c_ph=rng.uniform(0.02, 0.06),
        Cl0=rng.uniform(180.0, 320.0),
        alphaT=rng.uniform(0.014, 0.024),
        betapH=rng.uniform(0.05, 0.11),
        Kjump=rng.uniform(5.0, 12.0),
        gamma=rng.uniform(0.5, 1.5),
        qcurve=rng.uniform(0.7, 1.4),
        sigmaR=rng.uniform(0.03, 0.06),
        sigmaD=rng.uniform(0.006, 0.015),
    )


def cl_crit(T, pH, p):
    return p["Cl0"] * math.exp(-p["alphaT"] * (T - 25.0)) * (1.0 + p["betapH"] * (pH - 7.0))


def rate_true(Cl, T, pH, tex, p):
    Rp = (p["A0"]
          * math.exp(-p["EaR"] * (1.0 / (T + 273.15) - 1.0 / 298.15))
          * (1.0 + p["c_ph"] * (pH - 7.0) ** 2)
          * ((tex + 1.0) ** p["n_t"])
          * (1.0 + p["b_cl"] * Cl))
    cc = cl_crit(T, pH, p)
    excess = max(0.0, (Cl - cc) / cc)
    ec = excess ** p["qcurve"] if excess > 0.0 else 0.0
    return Rp * (1.0 + p["gamma"] * ec) * math.exp(p["Kjump"] * ec)


def margin_true(Cl, T, pH, p):
    cc = cl_crit(T, pH, p)
    return (cc - Cl) / cc


def gen_train(t, p, n):
    import random
    rng = random.Random(900 + t * 104729)
    rows = []
    for _ in range(n):
        T = rng.uniform(10.0, 70.0)
        pH = rng.uniform(5.0, 9.0)
        tex = rng.uniform(5.0, 500.0)
        cc = cl_crit(T, pH, p)
        Cl = rng.uniform(0.05 * cc, 0.85 * cc)
        Rt = rate_true(Cl, T, pH, tex, p)
        Robs = Rt * math.exp(rng.gauss(0.0, p["sigmaR"]))
        Dobs = margin_true(Cl, T, pH, p) + rng.gauss(0.0, p["sigmaD"])
        rows.append((Cl, T, pH, tex, Dobs, Robs))
    return rows


def gen_held(t, p, n):
    """HELD-OUT extrapolation set (never printed to the solver): three groups of
    n/3 rows each -- near-threshold, clearly active (same T,pH range as train),
    and outright untested chemistry (new hot/extreme-pH range), all regenerated
    deterministically from testId here only."""
    import random
    rng = random.Random(733999 + t * 15485863)
    rows = []
    for i in range(n):
        grp = i // (n // 3)
        if grp == 0:
            T = rng.uniform(10.0, 70.0); pH = rng.uniform(5.0, 9.0)
            cc = cl_crit(T, pH, p); Cl = rng.uniform(0.85 * cc, 1.05 * cc)
        elif grp == 1:
            T = rng.uniform(10.0, 70.0); pH = rng.uniform(5.0, 9.0)
            cc = cl_crit(T, pH, p); Cl = rng.uniform(1.15 * cc, 1.6 * cc)
        else:
            T = rng.uniform(70.0, 95.0)
            pH = rng.uniform(3.5, 4.8) if rng.random() < 0.5 else rng.uniform(9.2, 10.5)
            cc = cl_crit(T, pH, p); Cl = rng.uniform(1.1 * cc, 2.0 * cc)
        tex = rng.uniform(5.0, 500.0)
        Rt = rate_true(Cl, T, pH, tex, p)
        D = margin_true(Cl, T, pH, p)
        rows.append((Cl, T, pH, tex, D, Rt))
    return rows


# ---------- expression parsing / validation ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
)
_ALLOWED_VARS = {"Cl", "T", "pH", "tex", "D"}


def _validate_ast(tree):
    nodes = 0
    for node in ast.walk(tree):
        nodes += 1
        if not isinstance(node, _ALLOWED_NODES):
            return "disallowed syntax %s" % type(node).__name__
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS_ARITY):
                return "disallowed call"
            if node.keywords or len(node.args) != ALLOWED_FUNCS_ARITY[node.func.id]:
                return "bad function arity"
        if isinstance(node, ast.Name):
            nm = node.id
            if nm in ALLOWED_FUNCS_ARITY:
                continue
            if nm not in _ALLOWED_VARS:
                return "unknown name %s" % nm
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                return "non-numeric constant"
            v = float(node.value)
            if v != v or v in (float("inf"), float("-inf")):
                return "non-finite constant"
    if nodes > MAX_NODES:
        return "expression too large (%d nodes)" % nodes
    return None


SAFE_ENV_FUNCS = {
    "exp": lambda x: math.exp(max(-700.0, min(700.0, x))),
    "log": math.log,
    "sqrt": math.sqrt,
    "abs": abs,
    "max": max,
    "min": min,
}


def parse_expr(raw):
    text = raw.strip()
    if not text:
        fail("empty output")
    text = text.splitlines()[0].strip()   # single-expression contract: use first line only
    if not text:
        fail("empty first line")
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    err = _validate_ast(tree)
    if err:
        fail(err)
    try:
        code = compile(tree, "<expr>", "eval")
    except Exception:
        fail("compile error")
    return code


def eval_expr(code, Cl, T, pH, tex, D):
    env = dict(SAFE_ENV_FUNCS)
    env.update(Cl=Cl, T=T, pH=pH, tex=tex, D=D)
    glob = {"__builtins__": {}}
    try:
        v = eval(code, glob, env)
    except Exception:
        return None
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return None
    v = float(v)
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            header = fh.readline().split()
        n_train = int(header[0]); t = int(header[1])
    except Exception:
        fail("bad instance header")
    if t < 1 or t > 1000000 or n_train < 1:
        fail("bad test id")

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace")

    code = parse_expr(text)

    p = hidden_params(t)
    train = gen_train(t, p, n_train)
    held = gen_held(t, p, N_HELD)

    # checker's own baseline: constant = geometric mean of the SAME training rates
    log_mean = sum(math.log(r[5]) for r in train) / len(train)
    const = math.exp(log_mean)

    errs = []
    for (Cl, T, pH, tex, D, R) in held:
        pred = eval_expr(code, Cl, T, pH, tex, D)
        if pred is None or pred <= 0.0:
            fail("non-finite or non-positive prediction on a held-out environment")
        errs.append(abs(math.log10(pred) - math.log10(R)))
    mae = sum(errs) / len(errs)

    base_errs = [abs(math.log10(const) - math.log10(r[5])) for r in held]
    mae_base = sum(base_errs) / len(base_errs)

    sc = min(850.0, 100.0 * mae_base / max(1e-9, mae))
    print("MAE=%.6f MAE_baseline=%.6f  Ratio: %.6f" % (mae, mae_base, sc / 1000.0))


if __name__ == "__main__":
    main()
