#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for antibiotic-resistance-forecast.

- Reads the test id from <in>'s header, then REGENERATES the whole instance
  (including the hidden fitness cost `c` and the true post-treatment
  trajectory) from that id alone, via the SAME construction gen.py uses.  The
  hidden fitness cost lives ONLY here (and in gen.py's private construction --
  it is never written to gen.py's stdout).
- Parses the participant's single-line closed-form expression over variable
  `t` (query time) using the functions exp/log/sqrt, evaluates it at the
  held-out query times (which lie strictly AFTER the visible pre-treatment
  window), and scores by mean absolute error against the true resistant
  frequency there, normalised against an internal context-free baseline.
"""
import sys, math, ast, random

EPS = 0.10
MAX_OUT_BYTES = 20000
MAX_EXPR_CHARS = 2000
MAX_NODES = 200

T0 = 20.0
T1 = 24.0
QUERY_FRACS = [0.03, 0.08, 0.15, 0.25, 0.38, 0.52, 0.68, 0.84, 1.0]

LADDER = [
    (0.30, 0.70, 0.020, 20),
    (0.35, 0.80, 0.025, 18),
    (0.90, 1.30, 0.025, 18),
    (1.20, 1.80, 0.030, 16),
    (1.50, 2.20, 0.030, 16),
    (1.80, 2.50, 0.035, 14),
    (2.20, 3.00, 0.035, 14),
    (2.50, 3.50, 0.045, 12),
    (3.00, 4.00, 0.045, 12),
    (3.50, 4.60, 0.050, 10),
]

ALLOWED_FUNCS = {"exp": math.exp, "log": math.log, "sqrt": math.sqrt}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden instance (identical construction to gen.py) ----------
def build_instance(test_id):
    lo_idx = max(0, min(len(LADDER) - 1, test_id - 1))
    mult_lo, mult_hi, noise_sd, n_train = LADDER[lo_idx]
    rng = random.Random(90173 + test_id * 7919)

    for _attempt in range(500):
        mu = rng.uniform(1.0e-3, 4.0e-3)
        tau = 0.0 if rng.random() < 0.15 else rng.uniform(0.0, 3.0e-3)
        p0 = rng.uniform(0.03, 0.50)
        A = mu + tau
        c = A / p0
        alpha = rng.uniform(0.8, 2.2)
        d_thresh = c / alpha
        mult = rng.uniform(mult_lo, mult_hi)
        D = mult * d_thresh
        B = alpha * D - c
        R = A + B
        if abs(R) < 0.01:
            continue
        K0 = (A + B * p0) / (1.0 - p0)
        if K0 <= 0:
            continue

        query_times = [T0 + f * T1 for f in QUERY_FRACS]
        ok = True
        truths = []
        for t in query_times:
            M = K0 * math.exp(R * (t - T0))
            denom = M + B
            if abs(denom) < 1e-6:
                ok = False
                break
            pv = (M - A) / denom
            if not math.isfinite(pv) or not (0.0 <= pv <= 1.0):
                ok = False
                break
            truths.append(pv)
        if not ok:
            continue

        train_times = sorted(rng.uniform(0.5, T0 - 0.5) for _ in range(n_train))
        train_obs = []
        for _t in train_times:
            noisy = p0 + rng.gauss(0.0, noise_sd)
            train_obs.append(max(0.0, min(1.0, noisy)))

        return dict(mu=mu, tau=tau, alpha=alpha, D=D, T0=T0, T1=T1,
                    train_times=train_times, train_obs=train_obs,
                    query_times=query_times, truths=truths, p0=p0, c=c)
    raise RuntimeError("failed to build instance")


# ---------- safe expression parsing ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
)


def _validate(tree):
    nodes = 0
    for node in ast.walk(tree):
        nodes += 1
        if not isinstance(node, _ALLOWED_NODES):
            return None, "disallowed syntax %s" % type(node).__name__
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS):
                return None, "disallowed call"
            if node.keywords or len(node.args) != 1:
                return None, "bad function arity"
        if isinstance(node, ast.Name):
            if node.id != "t" and node.id not in ALLOWED_FUNCS:
                return None, "unknown name %s" % node.id
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                return None, "non-numeric constant"
            v = float(node.value)
            if not math.isfinite(v):
                return None, "non-finite constant"
    return nodes, None


def parse_expression(raw):
    text = raw.strip()
    if not text:
        fail("empty output")
    if len(text) > MAX_EXPR_CHARS:
        fail("expression too long")
    text = text.splitlines()[0].strip()
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    nodes, err = _validate(tree)
    if err:
        fail(err)
    if nodes > MAX_NODES:
        fail("expression too large (%d nodes)" % nodes)
    try:
        code = compile(tree, "<expr>", "eval")
    except Exception:
        fail("compile error")
    return code


def eval_at(code, tval):
    env = dict(ALLOWED_FUNCS)
    env["t"] = tval
    try:
        v = eval(code, {"__builtins__": {}}, env)
    except Exception:
        fail("evaluation error")
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        fail("non-numeric result")
    v = float(v)
    if not math.isfinite(v):
        fail("non-finite result")
    return v


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            first = fh.readline().split()
        t = int(first[0])
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

    code = parse_expression(text)

    inst = build_instance(t)
    preds = [eval_at(code, q) for q in inst["query_times"]]
    truths = inst["truths"]

    mae = sum(abs(p - y) for p, y in zip(preds, truths)) / len(truths)
    mae_baseline = sum(abs(0.5 - y) for y in truths) / len(truths)

    F = 1.0 / (mae + EPS)
    B = 1.0 / (mae_baseline + EPS)
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("mae=%.6f mae_baseline=%.6f  Ratio: %.6f" % (mae, mae_baseline, sc / 1000.0))


if __name__ == "__main__":
    main()
