#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the permafrost-thaw-forecast recovery task.

- Reads the test id from <in> (header), then regenerates the hidden station
  physics (latent-heat capacity LH, Stefan growth rate kappa, insulation
  feedback gamma, active-layer cap DMAX, baseline, noise) and the GRADED
  continuation of the forcing series entirely from that id.  The hidden law
  lives ONLY here (and in gen.py, identically).
- Parses the participant's STATEFUL predictor written in a tiny DSL:
      ACC <expr>      (optional, at most one; the energy-accumulator register)
      OUT <expr>       (required; the emitted ground-index prediction)
  Expressions are arithmetic over: the current forcing `f`, delayed forcing
  `fkJ` (= forcing J ticks ago), the current accumulator `A` (=`A0`), delayed
  accumulator `AkJ` (= accumulator J ticks ago), constants, + - * /, and the
  unary functions sig, step, relu, tanh, absv, sqrt.  ACC's own expression may
  reference AkJ (J>=1, i.e. the accumulator's OWN past) but never A/A0 (which
  is not yet defined for the current tick) -- this is how a solver builds a
  running, possibly-floored, integrator out of arithmetic primitives.
- The predictor is ROLLED forward over the GRADED window (registers reset to
  0 at the start of the roll -- no memory of the training span itself, only
  of whatever the solver's own baked-in constants encode about it), then
  scored by rollout MSE with a small node-count parsimony penalty:
      F = graded_MSE * (1 + LAMBDA * nodes)
      B = baseline_MSE * (1 + LAMBDA * 1)     # baseline = constant 0.0
      Ratio = min(1000, 100*B/F) / 1000
  A constant reproduces the baseline (~0.1).  Because the visible training
  window never leaves the frozen plateau, a predictor built only from the
  observed G values (ignoring energy accumulation) cannot see the capacity
  coming and stays low.  A predictor that integrates f the same way the
  physics does, and compares that running total to an estimated capacity,
  catches the early-crossing cases -- but the insulation feedback and the
  saturating cap are never visible in training, so even a good accumulator
  keeps real headroom below the ceiling.
"""
import sys, math, ast, random, re

LAMBDA = 0.006
DMAX = 3.0
N_HELD = 480
MAX_DELAY = 24
MAX_NODES = 80
MAX_OUT_BYTES = 200000

ALLOWED_FUNCS = {
    "sig": lambda x: 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x)))),
    "step": lambda x: 1.0 if x > 0 else 0.0,
    "relu": lambda x: x if x > 0 else 0.0,
    "tanh": math.tanh,
    "absv": abs,
    "sqrt": lambda x: math.sqrt(x) if x >= 0.0 else float("nan"),
}
_F_RE = re.compile(r"^fk(\d+)$")
_A_RE = re.compile(r"^Ak(\d+)$")


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden station physics (identical to gen.py) ----------
def params(t):
    rng = random.Random(9010013 + t * 7919)
    trend = rng.uniform(0.00060, 0.00110)
    cyc_amp1 = rng.uniform(0.42, 0.58)
    cyc_amp2 = rng.uniform(0.10, 0.18)
    per1 = rng.uniform(40, 70)
    per2 = rng.uniform(11, 19)
    ph1 = rng.uniform(0, 6.283185)
    ph2 = rng.uniform(0, 6.283185)
    proc_noise = rng.uniform(0.02, 0.035)
    kappa = rng.uniform(0.45, 1.00)
    gamma = rng.uniform(0.15, 0.45)
    eta = rng.uniform(0.45, 1.00)
    obs_sigma = 0.02 + 0.004 * (t - 1)
    g_plateau = rng.uniform(0.25, 0.55)
    return dict(trend=trend, cyc_amp1=cyc_amp1, cyc_amp2=cyc_amp2, per1=per1,
                per2=per2, ph1=ph1, ph2=ph2, proc_noise=proc_noise, kappa=kappa,
                gamma=gamma, eta=eta, obs_sigma=obs_sigma, g_plateau=g_plateau)


FRACS = [0.75, 0.85, 0.90, 0.65, 0.15, 0.20, 0.30, 0.60, 0.70, 0.25]


def n_train_for(t):
    return 300 - 6 * (t - 1)


def forcing_series(t, n, p):
    rng = random.Random(55010 + t * 104729)
    f = []
    for i in range(1, n + 1):
        v = (p['cyc_amp1'] * math.sin(2 * math.pi * i / p['per1'] + p['ph1'])
             + p['cyc_amp2'] * math.sin(2 * math.pi * i / p['per2'] + p['ph2'])
             + p['trend'] * i
             + rng.gauss(0.0, p['proc_noise']))
        f.append(v)
    return f


def hidden_capacity(t, f, n_train):
    total = len(f)
    E, Es_raw = 0.0, []
    for x in f:
        E = max(0.0, E + x)
        Es_raw.append(E)
    target = max(n_train + int(FRACS[t - 1] * N_HELD), n_train + 5)
    target = min(target, total)
    return Es_raw[target - 1]


def simulate(t, f, n_train, p, LH):
    total = len(f)
    rng = random.Random(770013 + t * 31)
    E = 0.0
    D = 0.0
    G = []
    for idx in range(total):
        amp = 1.0 + p['gamma'] * (D / DMAX)
        E = max(0.0, E + f[idx] * amp)
        if E >= LH:
            draw = p['kappa'] * math.sqrt(max(0.0, E - LH))
            D = DMAX * math.tanh(draw / DMAX)
        else:
            D = 0.0
        g = p['g_plateau'] + p['eta'] * D + rng.gauss(0.0, p['obs_sigma'])
        G.append(g)
    return G


# ---------- DSL parsing / validation ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
)


def _validate_ast(tree, allow_a0):
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return None, "disallowed syntax %s" % type(node).__name__
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS):
                return None, "disallowed call"
            if node.keywords or len(node.args) != 1:
                return None, "bad function arity"
        if isinstance(node, ast.Name):
            nm = node.id
            if nm in ALLOWED_FUNCS:
                continue
            if nm == "f":
                pass
            elif _F_RE.match(nm):
                if int(_F_RE.match(nm).group(1)) > MAX_DELAY:
                    return None, "forcing delay too large"
            elif nm in ("A", "A0"):
                if not allow_a0:
                    return None, "A/A0 not allowed in ACC's own expression"
            elif _A_RE.match(nm):
                if int(_A_RE.match(nm).group(1)) > MAX_DELAY:
                    return None, "accumulator delay too large"
                if int(_A_RE.match(nm).group(1)) < 1:
                    return None, "Ak0 is not a name; use A or A0"
            else:
                return None, "unknown name %s" % nm
            names.add(nm)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                return None, "non-numeric constant"
            v = float(node.value)
            if v != v or v in (float("inf"), float("-inf")):
                return None, "non-finite constant"
    return names, None


def _count_nodes(tree):
    return sum(1 for nd in ast.walk(tree)
               if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant)))


def _compile_expr(text, allow_a0):
    text = text.strip()
    if not text:
        fail("empty sub-expression")
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    names, err = _validate_ast(tree, allow_a0)
    if err:
        fail(err)
    try:
        code = compile(tree, "<dsl>", "eval")
    except Exception:
        fail("compile error")
    return code, names, _count_nodes(tree)


def parse_program(raw):
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        fail("empty program")
    acc_code = out_code = None
    used = set()
    nodes = 0
    seen_acc = seen_out = False
    for ln in lines:
        head = ln.split(None, 1)
        kw = head[0].upper()
        rest = head[1] if len(head) > 1 else ""
        if kw == "ACC":
            if seen_acc:
                fail("multiple ACC statements")
            seen_acc = True
            acc_code, na, n1 = _compile_expr(rest, allow_a0=False)
            used |= na
            nodes += n1
        elif kw == "OUT":
            if seen_out:
                fail("multiple OUT statements")
            seen_out = True
            out_code, no, n2 = _compile_expr(rest, allow_a0=True)
            used |= no
            nodes += n2
        else:
            fail("unknown statement '%s'" % kw)
    if not seen_out:
        fail("missing OUT statement")
    if nodes > MAX_NODES:
        fail("program too large (%d nodes)" % nodes)
    return acc_code, out_code, used, nodes


# ---------- rollout ----------
def roll(acc_code, out_code, used, forcing):
    f_delays = sorted(int(_F_RE.match(nm).group(1)) for nm in used if _F_RE.match(nm))
    a_delays = sorted(int(_A_RE.match(nm).group(1)) for nm in used if _A_RE.match(nm))
    n = len(forcing)
    A_hist = []
    preds = []
    glob = {"__builtins__": {}}
    for t in range(n):
        env = dict(ALLOWED_FUNCS)
        env["f"] = forcing[t]
        for J in f_delays:
            env["fk%d" % J] = forcing[t - J] if t - J >= 0 else forcing[0]
        if acc_code is not None:
            for J in a_delays:
                env["Ak%d" % J] = A_hist[t - J] if t - J >= 0 else 0.0
            try:
                a = float(eval(acc_code, glob, env))
            except Exception:
                fail("evaluation error in ACC")
            if a != a or a in (float("inf"), float("-inf")):
                fail("non-finite ACC result")
        else:
            a = 0.0
        A_hist.append(a)
        env["A"] = a
        env["A0"] = a
        for J in a_delays:
            env["Ak%d" % J] = A_hist[t - J] if t - J >= 0 else 0.0
        try:
            p = eval(out_code, glob, env)
        except Exception:
            fail("evaluation error in OUT")
        if not isinstance(p, (int, float)) or isinstance(p, bool):
            fail("non-numeric OUT result")
        p = float(p)
        if p != p or p in (float("inf"), float("-inf")):
            fail("non-finite OUT result")
        preds.append(p)
    return preds


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

    acc_code, out_code, used, nodes = parse_program(text)

    # regenerate hidden station + graded continuation
    n_train = n_train_for(t)
    p = params(t)
    total = n_train + N_HELD
    f_full = forcing_series(t, total, p)
    LH = hidden_capacity(t, f_full, n_train)
    G_full = simulate(t, f_full, n_train, p, LH)
    f_graded = f_full[n_train:]
    G_graded = G_full[n_train:]

    preds = roll(acc_code, out_code, used, f_graded)

    se = sum((pv - gv) ** 2 for pv, gv in zip(preds, G_graded))
    F_mse = se / len(G_graded)
    B_mse = sum(gv * gv for gv in G_graded) / len(G_graded)   # baseline: constant 0.0

    B = B_mse * (1.0 + LAMBDA * 1)
    F = F_mse * (1.0 + LAMBDA * nodes)
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("graded_MSE=%.6f baseline_MSE=%.6f nodes=%d  Ratio: %.6f"
          % (F_mse, B_mse, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
