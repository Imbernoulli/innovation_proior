#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the worn-clocktower backlash-gear identification task.

- Reads the test id from <in> (header), then regenerates the hidden gear train
  (small-integer tooth ratio r = p/q, backlash half-width D) and the HELD-OUT
  FAST, MANY-REVERSAL drive entirely from that id. The hidden law lives ONLY
  here (and, duplicated, in gen.py -- never printed on either stdout).
- Parses the participant's STATEFUL predictor written in a tiny DSL:
      STATE <expr>      (optional; defines the contact register S each tick)
      OUT   <expr>       (required; the emitted output-arbor angle)
  Expressions are arithmetic over: the current input angle `x`, delayed input
  angles `xkJ` (= input J ticks ago), the current contact state `S` (=`S0`),
  delayed contact states `SkJ` (= state J ticks ago), numeric constants,
  + - * /, the unary functions sig/step/relu/tanh/absv, and the binary
  functions min2(a,b)/max2(a,b). The STATE expression may reference `SkJ`
  (J>=1, i.e. past state) but never `S`/`S0` (no same-tick self-reference).
- The predictor is ROLLED forward on the held-out drive (state carried across
  time; missing history taps default to 0.0 at t=0, matching the generator's
  own fallback), then scored by held-out rollout MSE with a small node-count
  parsimony penalty (minimisation):
      F = heldout_MSE * (1 + LAMBDA * nodes)
      B = baseline_MSE * (1 + LAMBDA * 1)     # baseline = constant 0
      Ratio = min(1000, 100*B/F) / 1000
  A constant-zero predictor reproduces the baseline (~0.1). A memoryless
  curve of `x` alone (even a well-fit ratio) cannot represent the alternating
  backlash lag, so on a FAST reversing drive its error stays high. Recovering
  BOTH the small-rational ratio and the deadband state pulls MSE down a lot
  -- but sensor noise keeps even a perfect model off the ceiling, leaving
  headroom.
"""
import sys, math, ast, random, re

LAMBDA = 0.01
NH = 380
MAX_DELAY = 24
MAX_NODES = 80
MAX_OUT_BYTES = 200000

RATIO_CHOICES = [
    (2, 3), (3, 2), (3, 4), (4, 3), (2, 5), (5, 2), (3, 5), (5, 3),
    (4, 5), (5, 4), (5, 6), (6, 5), (5, 7), (7, 5), (2, 7), (7, 2),
    (3, 7), (7, 3), (4, 7), (7, 4), (5, 8), (8, 5), (3, 8), (8, 3),
]
AMP_LO, AMP_HI = 0.70, 1.00
AMP_REF = 0.85       # fixed reference swing used to scale noise/quantum (NOT the random per-test amp)
SIGMA_FRAC = 0.18    # sensor-noise std, as a fraction of r*AMP_REF
QSTEP_FRAC = 0.08    # escapement quantum, as a fraction of r*AMP_REF

UNARY_FUNCS = {
    "sig": lambda x: 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x)))),
    "step": lambda x: 1.0 if x > 0 else 0.0,
    "relu": lambda x: x if x > 0 else 0.0,
    "tanh": math.tanh,
    "absv": abs,
}
BINARY_FUNCS = {
    "min2": lambda a, b: a if a < b else b,
    "max2": lambda a, b: a if a > b else b,
}
_IN_RE = re.compile(r"^xk(\d+)$")
_ST_RE = re.compile(r"^Sk(\d+)$")


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden gear train (identical to gen.py) ----------
def hidden_params(t):
    rng = random.Random(900037 + t * 7919)
    p, q = rng.choice(RATIO_CHOICES)
    D = rng.uniform(0.15, 0.45)
    return p, q, D


def simulate(xs, r, D, sigma, qstep, seed):
    rng = random.Random(seed)
    c_prev = 0.0
    ys = []
    for x in xs:
        lo, hi = x - D, x + D
        c = hi if c_prev > hi else (lo if c_prev < lo else c_prev)
        raw = r * c
        if qstep > 1e-12:
            raw = round(raw / qstep) * qstep
        ys.append(raw + rng.gauss(0.0, sigma))
        c_prev = c
    return ys


def fast_drive(t, n):
    """Held-out FAST, many-reversal escapement wag; regenerated here only."""
    rng = random.Random(778121 + t * 15485863)
    per1 = rng.uniform(8.0, 14.0)
    per2 = rng.uniform(20.0, 30.0)
    ph1 = rng.uniform(0.0, 6.283185)
    ph2 = rng.uniform(0.0, 6.283185)
    amp = rng.uniform(AMP_LO, AMP_HI)
    xs = []
    for i in range(n):
        v = 0.75 * amp * math.sin(2 * math.pi * i / per1 + ph1) \
            + 0.25 * amp * math.sin(2 * math.pi * i / per2 + ph2)
        xs.append(v)
    return xs


# ---------- DSL parsing / validation ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
)


def _validate_ast(tree, allow_state):
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return None, "disallowed syntax %s" % type(node).__name__
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                return None, "disallowed call"
            fn = node.func.id
            if fn in UNARY_FUNCS:
                if node.keywords or len(node.args) != 1:
                    return None, "bad unary arity for %s" % fn
            elif fn in BINARY_FUNCS:
                if node.keywords or len(node.args) != 2:
                    return None, "bad binary arity for %s" % fn
            else:
                return None, "disallowed call %s" % fn
        if isinstance(node, ast.Name):
            nm = node.id
            if nm in UNARY_FUNCS or nm in BINARY_FUNCS:
                continue
            if nm == "x":
                pass
            elif _IN_RE.match(nm):
                if int(_IN_RE.match(nm).group(1)) > MAX_DELAY:
                    return None, "input delay too large"
            elif nm in ("S", "S0") or _ST_RE.match(nm):
                if not allow_state:
                    return None, "state reference not allowed here"
                if _ST_RE.match(nm) and int(_ST_RE.match(nm).group(1)) > MAX_DELAY:
                    return None, "state delay too large"
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


def _compile_expr(text, allow_state):
    text = text.strip()
    if not text:
        fail("empty sub-expression")
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    names, err = _validate_ast(tree, allow_state)
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
    state_code = out_code = None
    used = set()
    nodes = 0
    seen_state = seen_out = False
    for ln in lines:
        head = ln.split(None, 1)
        kw = head[0].upper()
        rest = head[1] if len(head) > 1 else ""
        if kw == "STATE":
            if seen_state:
                fail("multiple STATE statements")
            seen_state = True
            # STATE may reference x/xkJ/SkJ (J>=1) but NOT S/S0 (no same-tick self ref).
            state_code, ns, n1 = _compile_expr(rest, allow_state=True)
            if "S" in ns or "S0" in ns:
                fail("STATE may not reference S/S0 (same-tick self-reference)")
            used |= ns
            nodes += n1
        elif kw == "OUT":
            if seen_out:
                fail("multiple OUT statements")
            seen_out = True
            out_code, no, n3 = _compile_expr(rest, allow_state=True)
            used |= no
            nodes += n3
        else:
            fail("unknown statement '%s'" % kw)
    if not seen_out:
        fail("missing OUT statement")
    if nodes > MAX_NODES:
        fail("program too large (%d nodes)" % nodes)
    return state_code, out_code, used, nodes


# ---------- rollout ----------
def roll(state_code, out_code, used, xs):
    in_delays = sorted(int(_IN_RE.match(nm).group(1)) for nm in used if _IN_RE.match(nm))
    st_delays = sorted(int(_ST_RE.match(nm).group(1)) for nm in used if _ST_RE.match(nm))
    n = len(xs)
    S = []
    preds = []
    glob = {"__builtins__": {}}
    for t in range(n):
        env = dict(UNARY_FUNCS)
        env.update(BINARY_FUNCS)
        env["x"] = xs[t]
        for J in in_delays:
            env["xk%d" % J] = xs[t - J] if t - J >= 0 else xs[0]
        for J in st_delays:
            env["Sk%d" % J] = S[t - J] if t - J >= 0 else 0.0
        if state_code is not None:
            try:
                sv = eval(state_code, glob, env)
            except Exception:
                fail("evaluation error in STATE")
            if not isinstance(sv, (int, float)) or isinstance(sv, bool):
                fail("non-numeric STATE result")
            sv = float(sv)
            if sv != sv or sv in (float("inf"), float("-inf")):
                fail("non-finite STATE result")
        else:
            sv = 0.0
        S.append(sv)
        env["S"] = sv
        env["S0"] = sv
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

    state_code, out_code, used, nodes = parse_program(text)

    # regenerate hidden gear train + held-out fast reversing drive
    p, q, D = hidden_params(t)
    r = p / q
    sigma = SIGMA_FRAC * r * AMP_REF
    qstep = QSTEP_FRAC * r * AMP_REF
    xs = fast_drive(t, NH)
    y = simulate(xs, r, D, sigma, qstep, 777 + t * 17)

    preds = roll(state_code, out_code, used, xs)

    se = sum((pr - yv) ** 2 for pr, yv in zip(preds, y))
    F_mse = se / len(y)
    B_mse = sum((0.0 - yv) ** 2 for yv in y) / len(y)

    B = B_mse * (1.0 + LAMBDA * 1)
    F = F_mse * (1.0 + LAMBDA * nodes)
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("heldout_MSE=%.6f baseline_MSE=%.6f nodes=%d  Ratio: %.6f"
          % (F_mse, B_mse, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
