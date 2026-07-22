#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the lynx-hare delayed-crowding law-discovery task.

- Reads the test id from <in> (header), then regenerates the hidden ecosystem
  (delay tau, 4-season growth table r[], carrying capacity K) AND the HELD-OUT
  census entirely from that id.  The hidden law lives ONLY here.
- The held-out census is a large POPULATION CRASH: a perturbed initial condition
  far from the quiescent training regime, from which the true delayed system
  RINGS (overshoots and oscillates at the delay period) before settling.  It is
  never printed by gen.py.
- Parses the participant's one-step law -- a closed-form expression for the next
  density over the variables:
      x                 current density x[t]
      lag1..lag6        x[t-1] .. x[t-6]     (delayed densities)
      s                 current season index 0..3
      c0,c1,c2,c3       one-hot of the season (c_k = 1 iff s==k else 0)
  with constants and the operators + - * /, unary +/-, and absv/minv/maxv.
- The law is ROLLED FORWARD autoregressively on the held-out crash (its own
  predictions feed back as future lags), then scored by held-out rollout MSE with
  a small node-count parsimony penalty (minimisation):
      F = heldout_MSE * (1 + LAMBDA * nodes)
      B = baseline_MSE * (1 + LAMBDA * 1)      # baseline = persistence (x_next=x)
      Ratio = min(1000, 100*B/F) / 1000
  Persistence reproduces B (~0.1).  An undelayed logistic fit tracks the quiescent
  TRAIN census fine but predicts smooth relaxation where the true system rings, so
  it stays low on the crash.  A law that recovers the hidden delay + seasonal table
  drives MSE down -- but census noise plus estimation error keep even a good law
  well below the ceiling, leaving headroom.
"""
import sys, math, ast, random, re

S = 4
MAXLAG = 6
HIST = 8              # seeded history length (all lag refs valid from first step)
HOR = 70             # held-out rollout horizon
PROC = 0.03
SIG = 0.085          # held-out observation-noise floor (irreducible)
LAMBDA = 0.004
MAX_NODES = 400
MAX_OUT_BYTES = 200000
CLAMP = 1.0e4

ALLOWED_FUNCS = {
    "absv": abs,
    "minv": min,
    "maxv": max,
}
_LAG_RE = re.compile(r"^lag(\d+)$")
_ONEHOT_RE = re.compile(r"^c([0-3])$")


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden ecosystem (IDENTICAL to gen.py) ----------
def params(t):
    rng = random.Random(6050000 + t * 7919)
    plan = {1: (1, 1.30), 2: (1, 1.45), 3: (2, 1.42), 4: (2, 1.44), 5: (2, 1.54),
            6: (3, 1.32), 7: (3, 1.37), 8: (2, 1.50), 9: (2, 1.46), 10: (3, 1.35)}
    tau, R = plan[t]
    R += rng.uniform(-0.012, 0.012)
    amp = rng.uniform(0.06, 0.10)
    phase = rng.uniform(0.0, 2.0 * math.pi)
    K = rng.uniform(0.95, 1.10)
    crash = rng.uniform(0.30, 0.42)
    r = [R * (1.0 + amp * math.sin(2.0 * math.pi * s / S + phase)) for s in range(S)]
    return tau, R, K, r, crash


def true_next(x, r, K, tau, t):
    return x[t - 1] * r[t % S] * (1.0 - x[t - 1 - tau] / K)


def heldout(t):
    """Perturbed-crash held-out census; regenerated here only."""
    tau, R, K, r, crash = params(t)
    x = [0.5 * K] * (tau + 2)
    for _ in range(3000):
        x.append(true_next(x, r, K, tau, len(x)))
    xbar = sum(x[-8:]) / 8.0
    y = [crash * xbar] * HIST                # crashed flat initial history
    while len(y) < HIST + HOR:
        i = len(y)
        y.append(y[i - 1] * r[i % S] * (1.0 - y[i - 1 - tau] / K))
    clean = list(y)
    rng = random.Random(555 + t * 29)
    noisy = list(clean)
    for k in range(HIST, HIST + HOR):
        noisy[k] = clean[k] + rng.gauss(0.0, SIG)
    return clean, noisy


# ---------- expression parsing / validation ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
)


def _validate_ast(tree):
    used = set()
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return None, "disallowed syntax %s" % type(node).__name__
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS):
                return None, "disallowed call"
            if node.keywords:
                return None, "keyword args not allowed"
            fn = node.func.id
            if fn == "absv" and len(node.args) != 1:
                return None, "absv takes 1 arg"
            if fn in ("minv", "maxv") and len(node.args) != 2:
                return None, "minv/maxv take 2 args"
        if isinstance(node, ast.Name):
            nm = node.id
            if nm in ALLOWED_FUNCS:
                continue
            if nm == "x" or nm == "s" or _ONEHOT_RE.match(nm):
                pass
            elif _LAG_RE.match(nm):
                j = int(_LAG_RE.match(nm).group(1))
                if j < 1 or j > MAXLAG:
                    return None, "lag index out of range"
            else:
                return None, "unknown name %s" % nm
            used.add(nm)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                return None, "non-numeric constant"
            v = float(node.value)
            if v != v or v in (float("inf"), float("-inf")):
                return None, "non-finite constant"
    return used, None


def _count_nodes(tree):
    return sum(1 for nd in ast.walk(tree)
               if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant)))


def parse_law(raw):
    # accept the whole output; take the last non-empty logical line, strip an
    # optional leading "OUT" / "x_next =" / "=" decoration, then parse as expr.
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        fail("empty output")
    text = lines[-1]
    low = text.lower()
    for pre in ("x_next", "xnext", "out", "x[t+1]", "x'"):
        if low.startswith(pre):
            text = text[len(pre):].strip()
            break
    if text.startswith("="):
        text = text[1:].strip()
    if not text:
        fail("empty expression")
    if len(text) > MAX_OUT_BYTES:
        fail("expression too long")
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    used, err = _validate_ast(tree)
    if err:
        fail(err)
    nodes = _count_nodes(tree)
    if nodes > MAX_NODES:
        fail("law too large (%d nodes)" % nodes)
    try:
        code = compile(tree, "<law>", "eval")
    except Exception:
        fail("compile error")
    return code, used, nodes


# ---------- autoregressive rollout ----------
def roll(code, used, seq0):
    lag_needed = sorted(int(_LAG_RE.match(nm).group(1)) for nm in used if _LAG_RE.match(nm))
    seq = list(seq0)                     # seeded history, length HIST
    glob = {"__builtins__": {}}
    for i in range(HIST, HIST + HOR):
        s = i % S
        env = dict(ALLOWED_FUNCS)
        env["x"] = seq[i - 1]
        env["s"] = float(s)
        for k in range(4):
            env["c%d" % k] = 1.0 if s == k else 0.0
        for J in lag_needed:
            idx = i - 1 - J
            env["lag%d" % J] = seq[idx] if idx >= 0 else seq[0]
        try:
            p = eval(code, glob, env)
        except Exception:
            fail("evaluation error during rollout")
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            fail("non-numeric law result")
        p = float(p)
        if p != p or p in (float("inf"), float("-inf")):
            fail("non-finite law result")
        if p > CLAMP:
            p = CLAMP
        elif p < -CLAMP:
            p = -CLAMP
        seq.append(p)
    return seq


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            header = fh.readline().split()
        t = int(header[0])
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

    code, used, nodes = parse_law(text)

    clean, noisy = heldout(t)
    seq0 = clean[:HIST]                   # true crashed initial history (seed)
    pred = roll(code, used, seq0)

    F_mse = sum((pred[k] - noisy[k]) ** 2 for k in range(HIST, HIST + HOR)) / HOR
    # baseline: persistence (hold the last seeded value) == law "x"
    persist = seq0[-1]
    B_mse = sum((persist - noisy[k]) ** 2 for k in range(HIST, HIST + HOR)) / HOR

    B = B_mse * (1.0 + LAMBDA * 1)
    F = F_mse * (1.0 + LAMBDA * nodes)
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("heldout_MSE=%.6f baseline_MSE=%.6f nodes=%d  Ratio: %.6f"
          % (F_mse, B_mse, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
