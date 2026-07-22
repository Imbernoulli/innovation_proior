#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the Alternate-Bearing Orchard forecasting task.

- Reads the test id from <in> (header), then regenerates the ENTIRE hidden
  realised trajectory (train window + held-out late window) from that id
  alone -- exactly the same construction as gen.py.  The hidden recurrence
  parameter `a` (and the reserve invariant it implies) lives ONLY here.
- Parses the participant's ONE-LINE recurrence expression:
      OUT <expr>
  Expressions are arithmetic over `+ - * /`, parentheses, numeric constants,
  and delay-tap variables `x` (= the most recent known/predicted yield) and
  `xkJ` (= the yield J years further back), J in 1..MAX_DELAY.
- The expression is ROLLED FORWARD autoregressively over the held-out late
  window: it is seeded with the last MAX_DELAY+1 TRUE early-window (train)
  values, then from year T0 onward every tap is the program's OWN prior
  output (no further ground truth is ever fed in) -- a genuine multi-step
  extrapolation, not one-step-ahead prediction.
- Scored by held-out rollout MSE (against the true, noisy, late-window
  ledger) with a small node-count parsimony penalty, folded through a
  smooth saturating transform (maximisation of fidelity):
      Fpen = heldout_MSE * (1 + LAMBDA * nodes)
      Bpen = baseline_MSE * (1 + LAMBDA * 1)          (baseline: constant
             predictor = mean of the training years)
      r    = Bpen / Fpen
      Ratio = CAP * r / (r + (10*CAP - 1))            (CAP = 0.9)
  When Fpen == Bpen (r=1), Ratio = 0.1 exactly, so a constant predictor
  reproduces the baseline. Ratio increases monotonically with r but never
  reaches CAP even in the limit r -> infinity -- an EXACT held-out match
  (r -> infinity) still cannot saturate the score, so headroom above any
  reference solution is a structural guarantee, not a per-instance
  coincidence. The per-year process shock also keeps even the exact
  functional form from ever fully closing the gap.
"""
import sys, math, ast, random

LAMBDA = 0.01
MAX_DELAY = 6
MAX_NODES = 30
MAX_OUT_BYTES = 200000
CAP = 0.9

A_LO, A_HI = 0.6, 2.6
KICK_LO, KICK_HI = 1.6, 2.4
NOISE_CLIP = 0.35


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden recurrence (identical to gen.py) ----------
def hidden_params(t):
    rng = random.Random(900001 + t * 7919)
    a = rng.uniform(A_LO, A_HI)
    xstar = (1.0 + math.sqrt(1.0 + 4.0 * a)) / 2.0

    def kick():
        m = rng.uniform(KICK_LO, KICK_HI)
        return m if rng.random() < 0.5 else 1.0 / m

    x0 = xstar * kick()
    x1 = xstar * kick()
    return a, x0, x1


def sigma_proc(t):
    return 0.010 + 0.0015 * (t - 1)


def train_len(t):
    return 55 + 2 * (t - 1)


def extra_len(t):
    return 90 + 8 * (t - 1)


def full_trajectory(t):
    a, x0, x1 = hidden_params(t)
    sp = sigma_proc(t)
    n_total = train_len(t) + extra_len(t)
    rng = random.Random(31337 + t * 104729)
    xs = [x0, x1]
    for i in range(1, n_total - 1):
        raw = (a + xs[i]) / xs[i - 1]
        eps = rng.gauss(0.0, sp)
        eps = max(-NOISE_CLIP, min(NOISE_CLIP, eps))
        xs.append(raw * (1.0 + eps))
    return xs


# ---------- DSL parsing / validation ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
)


def _delay_of(name):
    """Return the delay J for 'x' (J=0) or 'xkJ' (J>=1), else None."""
    if name == "x":
        return 0
    if name.startswith("xk"):
        rest = name[2:]
        if rest.isdigit() and rest[0] != "0":
            return int(rest)
    return None


def _validate_ast(tree):
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            fail("disallowed syntax %s" % type(node).__name__)
        if isinstance(node, ast.Name):
            J = _delay_of(node.id)
            if J is None:
                fail("unknown name %s" % node.id)
            if J > MAX_DELAY:
                fail("delay too large in %s" % node.id)
            names.add(node.id)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                fail("non-numeric constant")
            v = float(node.value)
            if v != v or v in (float("inf"), float("-inf")):
                fail("non-finite constant")
    return names


def _count_nodes(tree):
    return sum(1 for nd in ast.walk(tree)
               if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Name, ast.Constant)))


def parse_program(raw):
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        fail("empty program")
    out_line = None
    for ln in lines:
        head = ln.split(None, 1)
        kw = head[0].upper()
        if kw != "OUT":
            fail("unknown statement '%s' (only OUT allowed)" % kw)
        if out_line is not None:
            fail("multiple OUT statements")
        out_line = head[1] if len(head) > 1 else ""
    if out_line is None:
        fail("missing OUT statement")
    text = out_line.strip()
    if not text:
        fail("empty expression")
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    names = _validate_ast(tree)
    nodes = _count_nodes(tree)
    if nodes > MAX_NODES:
        fail("program too large (%d nodes)" % nodes)
    try:
        code = compile(tree, "<dsl>", "eval")
    except Exception:
        fail("compile error")
    return code, names, nodes


# ---------- rollout ----------
def roll(code, names, seed_hist, n_extra):
    """seed_hist: list of the last (MAX_DELAY+1) TRUE early-window values, oldest first.
    Rolls the program forward n_extra steps, feeding its OWN outputs back in."""
    delays = sorted(_delay_of(nm) for nm in names)
    buf = list(seed_hist)  # buf[-1] = most recent known value = "x"
    glob = {"__builtins__": {}}
    preds = []
    for _ in range(n_extra):
        env = {}
        for J in delays:
            idx = len(buf) - 1 - J
            env["x" if J == 0 else "xk%d" % J] = buf[idx] if idx >= 0 else buf[0]
        try:
            p = eval(code, glob, env)
        except ZeroDivisionError:
            fail("division by zero during rollout")
        except Exception:
            fail("evaluation error in OUT")
        if not isinstance(p, (int, float)) or isinstance(p, bool):
            fail("non-numeric OUT result")
        p = float(p)
        if p != p or p in (float("inf"), float("-inf")):
            fail("non-finite OUT result")
        preds.append(p)
        buf.append(p)
    return preds


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            header = fh.readline().split()
            n_hdr = int(header[0])
            t = int(header[1])
            # `train` is parsed from the ACTUAL instance file -- i.e. exactly
            # the (rounded) values the participant received on stdin -- never
            # from an internally-regenerated higher-precision copy, so nobody
            # is scored against information they were not actually given.
            train = [float(fh.readline()) for _ in range(n_hdr)]
    except Exception:
        fail("bad instance header")
    if t < 1 or t > 100000:
        fail("bad test id")
    if len(train) != n_hdr or any(v != v or v in (float("inf"), float("-inf")) for v in train):
        fail("bad instance body")

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace")

    code, names, nodes = parse_program(text)

    T0 = train_len(t)
    Textra = extra_len(t)
    if T0 != n_hdr:
        fail("instance header mismatch")

    xs = full_trajectory(t)
    target = xs[T0:T0 + Textra]

    seed_hist = train[-(MAX_DELAY + 1):]
    if len(seed_hist) < MAX_DELAY + 1:
        seed_hist = [train[0]] * (MAX_DELAY + 1 - len(seed_hist)) + seed_hist

    preds = roll(code, names, seed_hist, Textra)

    se = sum((p - yv) ** 2 for p, yv in zip(preds, target))
    F_mse = se / len(target)

    mean_train = sum(train) / len(train)
    B_mse = sum((mean_train - yv) ** 2 for yv in target) / len(target)

    Fpen = F_mse * (1.0 + LAMBDA * nodes)
    Bpen = B_mse * (1.0 + LAMBDA * 1)
    r = Bpen / max(1e-12, Fpen)
    K = 10.0 * CAP - 1.0
    sc = CAP * r / (r + K)
    print("heldout_MSE=%.6f baseline_MSE=%.6f nodes=%d  Ratio: %.6f"
          % (F_mse, B_mse, nodes, sc))


if __name__ == "__main__":
    main()
