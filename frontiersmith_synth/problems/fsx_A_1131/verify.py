#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the "school in a current" drift-law / finger-index
task.

- Reads the test id from <in>'s header, then regenerates the hidden drift law
  (steady current + seasonal eddy + strengthening rip-current cubic) and the
  HELD-OUT / EXTRAPOLATION grading window entirely from that id.  The ground
  truth lives ONLY here (and, independently re-derived, in gen.py).
- Parses the participant's submission: a single line
      EXPR <expression>
  an arithmetic expression over the variable `t`, + - * / ** , parentheses,
  numeric constants, and the unary functions sin, cos, sqrt, abs, exp.
- Rolls a FINGER forward over the graded days T_train+1 .. T_train+H:
  each day the expression's (rounded, clamped) prediction is compared with
  the finger's current position; if they differ by more than the stated
  hysteresis band W the finger RELOCATES to the prediction (this is the only
  way the finger can move -- a pure function of the submitted law, exactly as
  a real fisherman could only ever act on his own model of the current, never
  on the fish's true position). Each day charges |finger-true_noisy|; a
  relocation beyond the free budget charges FEE_PER_EXCESS extra. The free
  budget is the number of relocations the TRUE (noiseless) trajectory would
  itself require under the same hysteresis band -- the physically-necessary
  count -- so only reactive/wasteful over-relocation is fined.
- Objective (minimise): F = total charge.  B = the same charge for the
  trivial "freeze the finger at the last known fix" policy.
      Ratio = min(1000, 100*B/max(1e-9,F)) / 1000
"""
import sys, math, ast, random

N_KEYSPACE = 1_000_000
SEED_LAW = 90176
SEED_TARGET = 727310231
MAX_OUT_BYTES = 20000
MAX_NODES = 120

ALLOWED_FUNCS = {
    "sin": math.sin, "cos": math.cos, "sqrt": math.sqrt,
    "abs": abs, "exp": math.exp,
}
_ALLOWED_AST = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
    ast.USub, ast.UAdd,
)


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden drift law (identical derivation to gen.py) ----------
def law_params(t):
    rng = random.Random(SEED_LAW + t * 7919)
    B0 = rng.uniform(0.30 * N_KEYSPACE, 0.45 * N_KEYSPACE)
    R = rng.uniform(70.0, 150.0)
    A_s = rng.uniform(2500.0, 7000.0)
    P = rng.uniform(35.0, 65.0)
    phase = rng.uniform(0.0, 2 * math.pi)
    gamma = rng.uniform(0.06, 0.14)
    T_train = 380 - 18 * (t - 1)
    H = 2 * T_train
    C3 = gamma * R / (T_train ** 2)
    sigma_obs = 250.0 + 40.0 * (t - 1)
    sigma_target = 0.35 * sigma_obs
    W = (3 * N_KEYSPACE) // 100
    return dict(N=N_KEYSPACE, B0=B0, R=R, A_s=A_s, P=P, phase=phase, C3=C3,
                T_train=T_train, H=H, sigma_obs=sigma_obs,
                sigma_target=sigma_target, W=W)


def law_value(t, p):
    return (p['B0'] + p['R'] * t
            + p['A_s'] * math.sin(2 * math.pi * t / p['P'] + p['phase'])
            + p['C3'] * (t ** 3))


def clampi(v, lo, hi):
    v = int(round(v))
    return lo if v < lo else (hi if v > hi else v)


# ---------- expression parsing / validation ----------
def _validate(tree):
    n = 0
    for node in ast.walk(tree):
        n += 1
        if not isinstance(node, _ALLOWED_AST):
            return None, "disallowed syntax %s" % type(node).__name__
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS):
                return None, "disallowed call"
            if node.keywords or len(node.args) != 1:
                return None, "bad function arity"
        if isinstance(node, ast.Name):
            if node.id in ALLOWED_FUNCS:
                pass
            elif node.id != "t":
                return None, "unknown name %s" % node.id
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                return None, "non-numeric constant"
            v = float(node.value)
            if v != v or v in (float("inf"), float("-inf")):
                return None, "non-finite constant"
            if abs(v) > 1e12:
                return None, "constant magnitude too large"
    return n, None


def parse_expr(raw):
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        fail("empty output")
    if len(lines) != 1:
        fail("expected exactly one non-blank line")
    line = lines[0]
    if not (line[:4].upper() == "EXPR"):
        fail("missing EXPR keyword")
    body = line[4:].strip()
    if not body:
        fail("empty expression")
    try:
        tree = ast.parse(body, mode="eval")
    except Exception:
        fail("parse error")
    nnodes, err = _validate(tree)
    if err:
        fail(err)
    if nnodes > MAX_NODES:
        fail("expression too large (%d nodes)" % nnodes)
    try:
        code = compile(tree, "<expr>", "eval")
    except Exception:
        fail("compile error")
    return code


def eval_expr(code, tval):
    glob = {"__builtins__": {}}
    env = dict(ALLOWED_FUNCS)
    env["t"] = float(tval)
    try:
        v = eval(code, glob, env)
    except Exception:
        fail("evaluation error at t=%d" % tval)
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        fail("non-numeric result at t=%d" % tval)
    v = float(v)
    if v != v or v in (float("inf"), float("-inf")):
        fail("non-finite prediction at t=%d" % tval)
    return v


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            header = fh.readline().split()
        T_train_in, t = int(header[0]), int(header[1])
    except Exception:
        fail("bad instance header")
    if t < 1 or t > 100000:
        fail("bad test id")

    p = law_params(t)
    if p['T_train'] != T_train_in:
        fail("instance/testid mismatch")

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace")

    code = parse_expr(text)

    N, W, T_train, H = p['N'], p['W'], p['T_train'], p['H']
    FEE_PER_EXCESS = 3.0 * W

    # graded (held-out / extrapolation) window truth, independent of submission
    rng_t = random.Random(SEED_TARGET + t * 911382323)
    true_series = []
    targets = []
    for i in range(H):
        day = T_train + 1 + i
        true_pos = law_value(day, p)
        true_series.append(clampi(true_pos, 0, N - 1))
        noisy = true_pos + rng_t.gauss(0.0, p['sigma_target'])
        targets.append(clampi(noisy, 0, N - 1))

    # free relocation budget = how many times a finger that saw the TRUE
    # (noiseless) trajectory would itself need to relocate under this same
    # hysteresis band. This is the physically-necessary relocation count;
    # only relocations beyond it are "reactive/wasteful" and charged.
    fbase = None
    ideal_reloc = 0
    for v in true_series:
        if fbase is None:
            fbase = v
        elif abs(v - fbase) > W:
            fbase = v
            ideal_reloc += 1
    free_budget = max(1, ideal_reloc)

    # candidate rollout: finger relocates only when its own prediction departs
    # from the current finger by more than the hysteresis band W
    finger = None
    relocations = 0
    F = 0.0
    for i in range(H):
        day = T_train + 1 + i
        pred = clampi(eval_expr(code, day), 0, N - 1)
        if finger is None:
            finger = pred
        elif abs(pred - finger) > W:
            finger = pred
            relocations += 1
        F += abs(finger - targets[i])

    extra = max(0, relocations - free_budget)
    F += FEE_PER_EXCESS * extra

    # baseline: freeze the finger at the last known (training-end) fix
    B_finger = clampi(law_value(T_train, p), 0, N - 1)
    B = sum(abs(B_finger - tg) for tg in targets)
    B = max(B, 1e-9)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("relocations=%d free_budget=%d F=%.4f B=%.4f  Ratio: %.6f"
          % (relocations, free_budget, F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
