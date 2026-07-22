#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the lazy-throttle-autopilot recovery task.

- Reads n_rows/testId from <in>'s header, then regenerates the hidden law
  (Kp, Ki, Umax, alpha) and the HELD-OUT, LARGE-STEP "stormy" setpoint
  profile entirely from testId.  The hidden law lives ONLY here (and in
  gen.py, never printed).
- Parses the participant's OUT expression: an arithmetic formula over the
  current error `e`, delayed errors `ekJ`, the running correction register
  `I` (auto-maintained: I[t] = I[t-1] + e[t], I[-1] = 0), delayed registers
  `IkJ`, constants, + - * /, and the unary functions sig, step, relu, tanh,
  absv.
- The expression is ROLLED forward on the held-out stormy setpoints (its own
  e/I registers evolve from its OWN simulated output, exactly like flying a
  real autopilot), then scored by rollout MSE against the TRUE hidden
  system's trajectory on the same setpoints, with a small node-count
  parsimony penalty (minimisation):
      F = heldout_MSE * (1 + LAMBDA * nodes)
      B = baseline_MSE * (1 + LAMBDA * 3)      # baseline: plain-P throttle
      Ratio = min(1000, 100*B/F) / 1000
  A plain proportional throttle law fit from the SAME calm log reproduces
  the baseline (~0.1).  An unclamped linear PI law fits the calm log just as
  well but has no notion of the throttle's limited authority, so on the
  stormy profile its windup runs away.  Recovering the tanh-shaped
  saturation (and its scale) from the faint calm-log curvature keeps the
  rollout close to the true trajectory -- but measurement noise plus the
  finite calm-log evidence keep even a good fit well below the ceiling,
  leaving headroom.
"""
import sys, math, ast, random, re

LAMBDA = 0.01
NH = 260
MAX_DELAY = 8
MAX_NODES = 40
MAX_OUT_BYTES = 20000

ALLOWED_FUNCS = {
    "sig": lambda x: 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x)))),
    "step": lambda x: 1.0 if x > 0 else 0.0,
    "relu": lambda x: x if x > 0 else 0.0,
    "tanh": math.tanh,
    "absv": abs,
}
_E_RE = re.compile(r"^ek(\d+)$")
_I_RE = re.compile(r"^Ik(\d+)$")


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden lazy-throttle law (identical to gen.py) ----------
def hidden_law(t):
    rng = random.Random(900001 + t * 7919)
    Kp = rng.uniform(0.6, 1.4)
    Ki = rng.uniform(0.05, 0.25)
    Umax = rng.uniform(3.0, 5.5)
    alpha = rng.uniform(0.15, 0.35)
    return Kp, Ki, Umax, alpha


def true_rollout(sp, Kp, Ki, Umax, alpha):
    n = len(sp)
    y = [0.0] * (n + 1)
    I = 0.0
    for t in range(n):
        e = sp[t] - y[t]
        I = I + e
        u = Umax * math.tanh((Kp * e + Ki * I) / Umax)
        y[t + 1] = y[t] + alpha * (u - y[t])
    return y[:n]


def stormy_setpoints(t, n):
    """Held-out LARGE-step profile (extrapolation regime); regenerated here only.
    Every held-out flight is meaningfully bigger than the calm training log (so the
    throttle's authority limit always matters to SOME degree); testId 7-10 are the
    genuinely severe "storm" cases where an unclamped linear extrapolation blows up."""
    rng = random.Random(918273 + t * 15485863)
    sp = []
    cur = 0.0
    steps_left = 0
    amp = 12.0 if t >= 7 else (5.0 + 0.4 * (t - 1))
    for _ in range(n):
        if steps_left <= 0:
            cur = rng.uniform(-amp, amp)
            steps_left = rng.randint(25, 45)
        sp.append(cur)
        steps_left -= 1
    return sp


# ---------- OUT-expression parsing / validation ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
)


def _validate_ast(tree):
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
            if nm == "e" or nm == "I":
                pass
            elif _E_RE.match(nm):
                if int(_E_RE.match(nm).group(1)) < 1 or int(_E_RE.match(nm).group(1)) > MAX_DELAY:
                    return None, "e-delay out of range"
            elif _I_RE.match(nm):
                if int(_I_RE.match(nm).group(1)) < 1 or int(_I_RE.match(nm).group(1)) > MAX_DELAY:
                    return None, "I-delay out of range"
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


def parse_program(raw):
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        fail("empty program")
    if len(lines) != 1:
        fail("program must be exactly one OUT line")
    ln = lines[0]
    head = ln.split(None, 1)
    if not head or head[0].upper() != "OUT" or len(head) != 2:
        fail("expected 'OUT <expr>'")
    text = head[1].strip()
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    names, err = _validate_ast(tree)
    if err:
        fail(err)
    try:
        code = compile(tree, "<dsl>", "eval")
    except Exception:
        fail("compile error")
    nodes = _count_nodes(tree)
    if nodes > MAX_NODES:
        fail("program too large (%d nodes)" % nodes)
    return code, names, nodes


def roll(out_code, used, sp, alpha):
    e_delays = sorted(int(_E_RE.match(nm).group(1)) for nm in used if _E_RE.match(nm))
    i_delays = sorted(int(_I_RE.match(nm).group(1)) for nm in used if _I_RE.match(nm))
    n = len(sp)
    y = [0.0] * (n + 1)
    e_hist = []
    I_hist = []
    glob = {"__builtins__": {}}
    I = 0.0
    for t in range(n):
        e = sp[t] - y[t]
        I = I + e
        env = dict(ALLOWED_FUNCS)
        env["e"] = e
        env["I"] = I
        for J in e_delays:
            env["ek%d" % J] = e_hist[t - J] if t - J >= 0 else 0.0
        for J in i_delays:
            env["Ik%d" % J] = I_hist[t - J] if t - J >= 0 else 0.0
        try:
            u = eval(out_code, glob, env)
        except Exception:
            fail("evaluation error in OUT")
        if not isinstance(u, (int, float)) or isinstance(u, bool):
            fail("non-numeric OUT result")
        u = float(u)
        if u != u or u in (float("inf"), float("-inf")):
            fail("non-finite OUT result")
        y[t + 1] = y[t] + alpha * (u - y[t])
        if y[t + 1] != y[t + 1] or y[t + 1] in (float("inf"), float("-inf")):
            fail("non-finite plant state")
        e_hist.append(e)
        I_hist.append(I)
    return y[:n]


def fit_P_only(sp, y, u):
    """Baseline: plain proportional throttle law fit from the calm log (no memory)."""
    num = 0.0
    den = 0.0
    for spv, yv, uv in zip(sp, y, u):
        e = spv - yv
        num += e * uv
        den += e * e
    return num / den if den > 1e-9 else 0.0


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            lines = fh.read().split("\n")
        header = lines[0].split()
        n_rows = int(header[0])
        t = int(header[1])
        if n_rows <= 0 or n_rows > 5000:
            fail("bad n_rows")
        sp_tr = [0.0] * n_rows
        y_tr = [0.0] * n_rows
        u_tr = [0.0] * n_rows
        for i in range(n_rows):
            parts = lines[1 + i].split()
            sp_tr[i] = float(parts[0])
            y_tr[i] = float(parts[1])
            u_tr[i] = float(parts[2])
    except Exception:
        fail("bad instance file")
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

    out_code, used, nodes = parse_program(text)

    # regenerate hidden law + held-out stormy setpoints
    Kp, Ki, Umax, alpha = hidden_law(t)
    sp_h = stormy_setpoints(t, NH)
    y_true_h = true_rollout(sp_h, Kp, Ki, Umax, alpha)

    y_sub_h = roll(out_code, used, sp_h, alpha)

    se = sum((a - b) ** 2 for a, b in zip(y_sub_h, y_true_h))
    F_mse = se / NH

    Kp0 = fit_P_only(sp_tr, y_tr, u_tr)

    # baseline rollout: plain-P law u = Kp0 * e, unclamped, no memory
    yb = [0.0] * (NH + 1)
    for tt in range(NH):
        eb = sp_h[tt] - yb[tt]
        ub = Kp0 * eb
        yb[tt + 1] = yb[tt] + alpha * (ub - yb[tt])
    B_mse = sum((a - b) ** 2 for a, b in zip(yb[:NH], y_true_h)) / NH

    B = B_mse * (1.0 + LAMBDA * 3)
    F = F_mse * (1.0 + LAMBDA * nodes)
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("heldout_MSE=%.6f baseline_MSE=%.6f nodes=%d  Ratio: %.6f"
          % (F_mse, B_mse, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
