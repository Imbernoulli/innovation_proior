#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for "River flow after the snow is gone".

- Reads the test id from <in> (header), then regenerates the hidden watershed
  (freeze threshold, melt rate, moisture decay/coupling, quickflow/meltflow
  coefficients, baseflow) and the HELD-OUT winter-into-spring drive entirely
  from that id.  The hidden law lives ONLY here (and duplicated verbatim from
  gen.py -- never imported, never printed to the participant).
- Parses the participant's STATEFUL predictor written in a tiny DSL:
      STORE <accum_expr>      (optional; at most one -- updates the ONE
                                allowed storage register)
      OUT   <out_expr>         (required; the emitted flow value)
  Expressions are arithmetic over: the current precip `p`, delayed precip
  `pkJ` (J ticks ago), the current temperature `tm`, delayed temperature
  `tmkJ`, the storage register `SW` (=`SW0`, this tick's value -- OUT only),
  delayed storage `SWkJ` (J ticks ago, both STORE and OUT), constants,
  + - * /, and the unary functions sig, step, relu, tanh, absv.
  `STORE`'s accum_expr may reference `p`, `tm`, their delayed taps and
  `SWkJ` (J>=1) -- but NEVER `SW`/`SW0` (the value it is about to produce).
- Each tick the grader updates the register
      SW[t] = clip(SW[t-1] + accum_expr(t), 0, CAP)
  (no STORE line => SW stays 0 forever), then evaluates OUT to produce the
  predicted flow, and rolls this forward over the held-out season with state
  carried across time.
- Scored by held-out MSE with a small node-count parsimony penalty
  (maximisation of predictive skill, expressed as an error ratio):
      F = heldout_MSE * (1 + LAMBDA * nodes)
      B = baseline_MSE * (1 + LAMBDA * 1)   # baseline = constant mean(train flow)
      Ratio = min(1000, 100*B/F) / 1000
  A constant predictor reproduces the baseline (~0.1).  A memoryless
  rain-reacts-to-rain fit explains most of the RAIN-season training variance
  but has no notion of a persisting snowpack, so it stays low on the
  snowmelt-dominated held-out season.  Recovering the storage mechanism (and
  the antecedent-moisture amplification) drives MSE down -- but sensor noise
  and the unmodelled interactions keep even a good model below the ceiling,
  leaving headroom.
"""
import sys, math, ast, random, re

LAMBDA = 0.010
CAP = 8.0
N_ACC = 160
N_MELT = 140
N_DRY = 100
NH = N_ACC + N_MELT + N_DRY
MAX_DELAY = 24
MAX_NODES = 140
MAX_OUT_BYTES = 200000

ALLOWED_FUNCS = {
    "sig": lambda x: 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x)))),
    "step": lambda x: 1.0 if x > 0 else 0.0,
    "relu": lambda x: x if x > 0 else 0.0,
    "tanh": math.tanh,
    "absv": abs,
}
_PK_RE = re.compile(r"^pk(\d+)$")
_TMK_RE = re.compile(r"^tmk(\d+)$")
_SWK_RE = re.compile(r"^SWk(\d+)$")


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden watershed (identical to gen.py) ----------
def hidden_params(t):
    rng = random.Random(9130001 + t * 7919)
    Tf = rng.uniform(-0.05, 0.05)
    k_melt = rng.uniform(0.35, 0.55)
    gamma = rng.uniform(0.55, 0.75)
    eta = rng.uniform(0.30, 0.60)
    alpha_r = rng.uniform(0.55, 0.80)
    kappa = rng.uniform(0.80, 1.60)
    alpha_m = rng.uniform(0.70, 0.95)
    b0 = rng.uniform(0.05, 0.09)
    return Tf, k_melt, gamma, eta, alpha_r, kappa, alpha_m, b0


def simulate(precip, temp, params, sigma, seed):
    Tf, k_melt, gamma, eta, alpha_r, kappa, alpha_m, b0 = params
    rng = random.Random(seed)
    sw = 0.0
    am = 0.0
    flow = []
    for i in range(len(precip)):
        pv, tv = precip[i], temp[i]
        snow_in = pv if tv < Tf else 0.0
        rain_in = pv if tv >= Tf else 0.0
        melt_potential = k_melt * max(0.0, tv - Tf)
        melt = min(sw, melt_potential)
        sw = min(CAP, sw + snow_in - melt)
        am_prev = am
        am = gamma * am_prev + rain_in + eta * melt
        quick = alpha_r * rain_in * (1.0 + kappa * am_prev)
        meltflow = alpha_m * melt
        q = b0 + quick + meltflow + rng.gauss(0.0, sigma)
        flow.append(max(0.0, q))
    return flow


def train_weather(t, n):
    """Mild, mostly-above-freezing RAIN-season weather; a few brief, shallow
    cold snaps that build only a little transient snow (just enough to hint
    the storage mechanism exists, never enough to look like a real winter)."""
    rng = random.Random(55001 + t * 104729)
    per1 = rng.uniform(70, 110)
    per2 = rng.uniform(150, 240)
    ph1 = rng.uniform(0, 6.283185)
    ph2 = rng.uniform(0, 6.283185)
    ndips = rng.randint(2, 3)
    centers = sorted(rng.uniform(0.05 * n, 0.95 * n) for _ in range(ndips))
    depths = [rng.uniform(0.40, 0.60) for _ in range(ndips)]
    widths = [rng.uniform(4, 9) for _ in range(ndips)]
    temp = []
    for i in range(n):
        v = 0.30 + 0.15 * math.sin(2 * math.pi * i / per1 + ph1) \
                 + 0.05 * math.sin(2 * math.pi * i / per2 + ph2)
        for c, dp, wd in zip(centers, depths, widths):
            v -= dp * math.exp(-0.5 * ((i - c) / wd) ** 2)
        v += rng.gauss(0.0, 0.02)
        temp.append(v)
    precip = []
    for i in range(n):
        if rng.random() < 0.20:
            precip.append(min(1.4, rng.gammavariate(2.0, 0.18)))
        else:
            precip.append(0.0)
    return precip, temp


def heldout_weather(t, n_acc, n_melt, n_dry):
    """HELD-OUT winter-into-spring drive: a long cold accumulation phase (many
    quiet snow-building weeks), a warming melt ramp with a few rain-on-snow
    bursts, then a sustained warm/dry tail AFTER the snowpack has fully run
    out (temperature stays high but there is no stored snow left to melt);
    regenerated here only."""
    rng = random.Random(710003 + t * 15485863)
    per_a = rng.uniform(30, 50)
    ph_a = rng.uniform(0, 6.283185)
    per_m = rng.uniform(18, 30)
    ph_m = rng.uniform(0, 6.283185)

    temp = []
    for i in range(n_acc):
        v = -0.35 + 0.10 * math.sin(2 * math.pi * i / per_a + ph_a) + rng.gauss(0.0, 0.05)
        temp.append(v)
    for j in range(n_melt):
        frac = j / max(1, n_melt - 1)
        v = -0.20 + 1.10 * frac + 0.12 * math.sin(2 * math.pi * j / per_m + ph_m) + rng.gauss(0.0, 0.05)
        temp.append(v)
    for j in range(n_dry):
        v = 0.75 + 0.10 * math.sin(2 * math.pi * j / 20 + ph_m) + rng.gauss(0.0, 0.05)
        temp.append(v)

    precip = []
    for i in range(n_acc):
        if rng.random() < 0.10:
            precip.append(min(1.4, rng.gammavariate(1.6, 0.14)))
        else:
            precip.append(0.0)
    for j in range(n_melt):
        if rng.random() < 0.12:
            precip.append(min(1.4, rng.gammavariate(2.0, 0.20)))
        else:
            precip.append(0.0)
    for j in range(n_dry):
        if rng.random() < 0.06:
            precip.append(min(1.4, rng.gammavariate(1.6, 0.15)))
        else:
            precip.append(0.0)
    return precip, temp


# ---------- DSL parsing / validation ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
)


def _validate_ast(tree, allow_sw0):
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
            if nm == "p" or nm == "tm":
                pass
            elif _PK_RE.match(nm):
                if int(_PK_RE.match(nm).group(1)) > MAX_DELAY:
                    return None, "precip delay too large"
            elif _TMK_RE.match(nm):
                if int(_TMK_RE.match(nm).group(1)) > MAX_DELAY:
                    return None, "temp delay too large"
            elif nm in ("SW", "SW0"):
                if not allow_sw0:
                    return None, "SW (zero-lag storage) not allowed in STORE (would self-reference)"
            elif _SWK_RE.match(nm):
                if int(_SWK_RE.match(nm).group(1)) > MAX_DELAY:
                    return None, "storage delay too large"
                if int(_SWK_RE.match(nm).group(1)) < 1:
                    return None, "SWk0 is not a valid name (use SW)"
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


def _compile_expr(text, allow_sw0):
    text = text.strip()
    if not text:
        fail("empty sub-expression")
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    names, err = _validate_ast(tree, allow_sw0)
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
    store_code = out_code = None
    used = set()
    nodes = 0
    seen_store = seen_out = False
    for ln in lines:
        head = ln.split(None, 1)
        kw = head[0].upper()
        rest = head[1] if len(head) > 1 else ""
        if kw == "STORE":
            if seen_store:
                fail("multiple STORE statements")
            seen_store = True
            store_code, ns, n1 = _compile_expr(rest, allow_sw0=False)
            used |= ns
            nodes += n1
        elif kw == "OUT":
            if seen_out:
                fail("multiple OUT statements")
            seen_out = True
            out_code, no, n2 = _compile_expr(rest, allow_sw0=True)
            used |= no
            nodes += n2
        else:
            fail("unknown statement '%s'" % kw)
    if not seen_out:
        fail("missing OUT statement")
    if nodes > MAX_NODES:
        fail("program too large (%d nodes)" % nodes)
    return store_code, out_code, used, nodes


# ---------- rollout ----------
def roll(store_code, out_code, used, precip, temp):
    p_delays = sorted(int(_PK_RE.match(nm).group(1)) for nm in used if _PK_RE.match(nm))
    tm_delays = sorted(int(_TMK_RE.match(nm).group(1)) for nm in used if _TMK_RE.match(nm))
    sw_delays = sorted(int(_SWK_RE.match(nm).group(1)) for nm in used if _SWK_RE.match(nm))
    n = len(precip)
    SW = []
    preds = []
    glob = {"__builtins__": {}}
    prev_sw = 0.0
    for t in range(n):
        env = dict(ALLOWED_FUNCS)
        env["p"] = precip[t]
        env["tm"] = temp[t]
        for J in p_delays:
            env["pk%d" % J] = precip[t - J] if t - J >= 0 else precip[0]
        for J in tm_delays:
            env["tmk%d" % J] = temp[t - J] if t - J >= 0 else temp[0]
        for J in sw_delays:
            env["SWk%d" % J] = SW[t - J] if t - J >= 0 else 0.0
        if store_code is not None:
            try:
                av = float(eval(store_code, glob, env))
            except Exception:
                fail("evaluation error in STORE")
            if av != av or av in (float("inf"), float("-inf")):
                fail("non-finite STORE value")
            new_sw = min(CAP, max(0.0, prev_sw + av))
        else:
            new_sw = 0.0
        SW.append(new_sw)
        prev_sw = new_sw
        env["SW"] = new_sw
        env["SW0"] = new_sw
        try:
            pr = eval(out_code, glob, env)
        except Exception:
            fail("evaluation error in OUT")
        if not isinstance(pr, (int, float)) or isinstance(pr, bool):
            fail("non-numeric OUT result")
        pr = float(pr)
        if pr != pr or pr in (float("inf"), float("-inf")):
            fail("non-finite OUT result")
        preds.append(pr)
    return preds


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            header = fh.readline().split()
        t = int(header[1])
        ntrain = int(header[0])
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

    store_code, out_code, used, nodes = parse_program(text)

    params = hidden_params(t)
    sigma = 0.018 + 0.003 * (t - 1)

    train_precip, train_temp = train_weather(t, ntrain)
    train_flow = simulate(train_precip, train_temp, params, sigma, 2231 + t * 13)
    cmean = sum(train_flow) / len(train_flow)

    held_precip, held_temp = heldout_weather(t, N_ACC, N_MELT, N_DRY)
    held_flow = simulate(held_precip, held_temp, params, sigma, 990001 + t * 17)

    preds = roll(store_code, out_code, used, held_precip, held_temp)

    se = sum((p - y) ** 2 for p, y in zip(preds, held_flow))
    F_mse = se / len(held_flow)
    B_mse = sum((cmean - y) ** 2 for y in held_flow) / len(held_flow)

    B = B_mse * (1.0 + LAMBDA * 1)
    F = F_mse * (1.0 + LAMBDA * nodes)
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("heldout_MSE=%.6f baseline_MSE=%.6f nodes=%d  Ratio: %.6f"
          % (F_mse, B_mse, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
