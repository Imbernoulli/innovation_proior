#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the antique-greenhouse-thermostat recovery task.

- Reads the test id from <in> (header), then regenerates the hidden thermostat
  (band [L,H], k-step + fractional actuation delay) and the HELD-OUT FAST,
  NON-MONOTONE drive entirely from that id.  The hidden law lives ONLY here.
- Parses the participant's STATEFUL predictor written in a tiny DSL:
      LATCH <set_expr> | <reset_expr>      (optional; Schmitt latch on the drive)
      OUT   <expr>                          (required; the emitted heater value)
  Expressions are arithmetic over: the current drive `d`, delayed drives `dkJ`
  (= drive J steps ago), the current latch `S` (=`S0`), delayed latches `SkJ`
  (= latch J steps ago), constants, + - * /, and the unary functions
  sig, step, relu, tanh, absv.  The LATCH conditions may reference the drive
  taps ONLY (never the latch itself).
- The predictor is ROLLED forward on the held-out drive (state carried across
  time), then scored by held-out rollout MSE with a small node-count parsimony
  penalty (minimisation):
      F = heldout_MSE * (1 + LAMBDA * nodes)
      B = baseline_MSE * (1 + LAMBDA * 1)     # baseline = constant 0.5
      Ratio = min(1000, 100*B/F) / 1000
  A constant reproduces the baseline (~0.1).  A memoryless static curve of `d`
  cannot represent the two hysteresis branches nor the lag, so it stays low.
  A latch-with-delay that recovers the hidden band and lag drives MSE down --
  but sensor noise plus the unmodelled FRACTIONAL lag keep even a good latch
  well below the ceiling, leaving headroom.
"""
import sys, math, ast, random, re

LAMBDA = 0.008
OFF, AMP = 0.2, 0.6
NH = 420
MAX_DELAY = 24
MAX_NODES = 80
MAX_OUT_BYTES = 200000

ALLOWED_FUNCS = {
    "sig": lambda x: 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x)))),
    "step": lambda x: 1.0 if x > 0 else 0.0,
    "relu": lambda x: x if x > 0 else 0.0,
    "tanh": math.tanh,
    "absv": abs,
}
_DRIVE_RE = re.compile(r"^dk(\d+)$")
_LATCH_RE = re.compile(r"^Sk(\d+)$")


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden thermostat (identical to gen.py) ----------
def params(t):
    rng = random.Random(4200191 + t * 7919)
    L = rng.uniform(0.34, 0.42)
    w = rng.uniform(0.14, 0.22)
    H = L + w
    k = rng.choice([2, 3])
    phi = rng.uniform(0.30, 0.55)
    return L, H, k, phi


def latch_roll(drive, L, H):
    s = 0
    out = []
    for d in drive:
        if d < L:
            s = 1
        elif d > H:
            s = 0
        out.append(s)
    return out


def true_output(drive, L, H, k, phi, sigma, seed):
    S = latch_roll(drive, L, H)
    rng = random.Random(seed)
    y = []
    for t in range(len(drive)):
        a = S[t - k] if t - k >= 0 else 0
        b = S[t - k - 1] if t - k - 1 >= 0 else 0
        y.append(OFF + AMP * ((1 - phi) * a + phi * b) + rng.gauss(0.0, sigma))
    return y


def fast_drive(t, n):
    """Held-out FAST non-monotone drive (extrapolation regime); regenerated here only."""
    rng = random.Random(20261 + t * 15485863)
    per1 = rng.uniform(40, 60)
    per2 = rng.uniform(14, 20)
    ph1 = rng.uniform(0, 6.283185)
    ph2 = rng.uniform(0, 6.283185)
    d = []
    for i in range(n):
        v = 0.46 + 0.36 * math.sin(2 * math.pi * i / per1 + ph1) \
                 + 0.09 * math.sin(2 * math.pi * i / per2 + ph2)
        d.append(min(1.0, max(0.0, v)))
    return d


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
            if not (isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS):
                return None, "disallowed call"
            if node.keywords or len(node.args) != 1:
                return None, "bad function arity"
        if isinstance(node, ast.Name):
            nm = node.id
            if nm in ALLOWED_FUNCS:
                continue
            if nm == "d":
                pass
            elif _DRIVE_RE.match(nm):
                if int(_DRIVE_RE.match(nm).group(1)) > MAX_DELAY:
                    return None, "drive delay too large"
            elif nm in ("S", "S0") or _LATCH_RE.match(nm):
                if not allow_state:
                    return None, "latch reference not allowed in LATCH condition"
                if _LATCH_RE.match(nm) and int(_LATCH_RE.match(nm).group(1)) > MAX_DELAY:
                    return None, "latch delay too large"
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
    if "|" in text:
        fail("stray '|' in expression")
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
    set_code = reset_code = out_code = None
    used = set()
    nodes = 0
    seen_latch = seen_out = False
    for ln in lines:
        head = ln.split(None, 1)
        kw = head[0].upper()
        rest = head[1] if len(head) > 1 else ""
        if kw == "LATCH":
            if seen_latch:
                fail("multiple LATCH statements")
            seen_latch = True
            if "|" not in rest:
                fail("LATCH needs 'set | reset'")
            a, b = rest.split("|", 1)
            set_code, na, n1 = _compile_expr(a, allow_state=False)
            reset_code, nb, n2 = _compile_expr(b, allow_state=False)
            used |= na | nb
            nodes += n1 + n2
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
    return set_code, reset_code, out_code, used, nodes


# ---------- rollout ----------
def roll(set_code, reset_code, out_code, used, drive):
    drive_delays = sorted(int(_DRIVE_RE.match(nm).group(1)) for nm in used if _DRIVE_RE.match(nm))
    latch_delays = sorted(int(_LATCH_RE.match(nm).group(1)) for nm in used if _LATCH_RE.match(nm))
    n = len(drive)
    S = []
    preds = []
    glob = {"__builtins__": {}}
    for t in range(n):
        env = dict(ALLOWED_FUNCS)
        env["d"] = drive[t]
        for J in drive_delays:
            env["dk%d" % J] = drive[t - J] if t - J >= 0 else drive[0]
        if set_code is not None:
            try:
                sv = float(eval(set_code, glob, env))
                rv = float(eval(reset_code, glob, env))
            except Exception:
                fail("evaluation error in LATCH")
            if sv != sv or rv != rv or sv in (float("inf"), float("-inf")) or rv in (float("inf"), float("-inf")):
                fail("non-finite LATCH condition")
            if sv > 0:
                s = 1.0
            elif rv > 0:
                s = 0.0
            else:
                s = S[-1] if S else 0.0
        else:
            s = 0.0
        S.append(s)
        env["S"] = s
        env["S0"] = s
        for J in latch_delays:
            env["Sk%d" % J] = S[t - J] if t - J >= 0 else 0.0
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

    set_code, reset_code, out_code, used, nodes = parse_program(text)

    # regenerate hidden thermostat + held-out fast drive
    L, H, k, phi = params(t)
    sigma = 0.05 + 0.004 * (t - 1)
    drive = fast_drive(t, NH)
    y = true_output(drive, L, H, k, phi, sigma, 777 + t * 17)

    preds = roll(set_code, reset_code, out_code, used, drive)

    se = sum((p - yv) ** 2 for p, yv in zip(preds, y))
    F_mse = se / len(y)
    B_mse = sum((0.5 - yv) ** 2 for yv in y) / len(y)

    B = B_mse * (1.0 + LAMBDA * 1)
    F = F_mse * (1.0 + LAMBDA * nodes)
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("heldout_MSE=%.6f baseline_MSE=%.6f nodes=%d  Ratio: %.6f"
          % (F_mse, B_mse, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
