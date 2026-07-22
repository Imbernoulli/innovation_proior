#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the mystery-pipeline cycle-count recovery task.  The
solver submits ONE closed-form expression predicting the stopwatch-measured
cycle count of a straight-line program from eight structural counts.

- Reads the case id from <in> (header) and the printed TRAIN table (used only
  to build the checker's own constant baseline -- the mean training cycle
  count).
- Regenerates the HELD-OUT probe set entirely from the case id: programs with
  much LONGER consecutive-MUL clusters than anything in the training table
  (the extrapolation split).  The hidden per-hazard costs and the multiply
  unit's contention law live ONLY here.
- Parses the submitted expression with a strict AST whitelist:
      names     n nA nM nL nS cLU cMF cST
      operators + - * / **  and unary +/-
      functions sqrt log exp sig tanh absv
      numeric constants
- Evaluates it on the held-out set, computes a bounded symmetric relative
  error per point, averages, and adds a small node-count parsimony penalty
  (maximisation of the resulting Ratio):
      metric = mean_i min(1, |p_i - t_i| / (|p_i| + |t_i| + eps))
      O = metric * (1 + LAMBDA * nodes)
      B = baseline_metric * (1 + LAMBDA * 1)      # baseline = constant mean(train cycles)
      Ratio = min(1000, 100 * B / O) / 1000
  A constant reproduces the baseline (~0.1).  An additive per-opcode /
  per-hazard-count LINEAR regression fits the training table (whose MUL
  clusters are always short) almost exactly, but the true multiply-unit
  contention grows like a triangular number of the cluster length -- a
  single-server queue whose service time exceeds its arrival interval
  backlogs superlinearly.  Only an expression that reproduces that
  triangular growth survives the long held-out clusters; noise plus the
  parsimony tax keep even a correct law below the ceiling, leaving headroom.
"""
import sys, math, ast, random

# ---- fixed design constants (mirrored byte-for-byte in gen.py) ----
LAMBDA = 0.002
CAP = 1.0
MAX_NODES = 80
MAX_OUT_BYTES = 100000

N_HELD = 100
N_HELD_LO, N_HELD_HI = 30, 70
P_DEP_HELD = 0.5
NOISE_HELD_SIGMA = 18.0

ALLOWED_FUNCS = {
    "sqrt": lambda x: math.sqrt(x),
    "log":  lambda x: math.log(x),
    "exp":  lambda x: math.exp(max(-700.0, min(700.0, x))),
    "sig":  lambda x: 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x)))),
    "tanh": math.tanh,
    "absv": abs,
}
ALLOWED_NAMES = {"n", "nA", "nM", "nL", "nS", "cLU", "cMF", "cST"}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden mechanistic law + held-out regime (identical to gen.py's train law) ----------
def true_cycles(n, cLU, cMF, cST):
    return n + 2 * cLU + cMF + cST * (cST + 1) // 2


def gen_program(rng, n, maxR, p_dep):
    R = rng.randint(0, maxR)
    if R > n:
        R = n
    op = [None] * n
    if R > 0:
        start = rng.randint(0, n - R)
        for i in range(start, start + R):
            op[i] = 'M'
    for i in range(n):
        if op[i] is None:
            op[i] = rng.choice(('A', 'L', 'S'))
    dep = [0] * n
    for j in range(1, n):
        prev, cur = op[j - 1], op[j]
        if prev == 'S':
            dep[j] = 0
        elif prev == 'M' and cur == 'M':
            dep[j] = 0
        else:
            dep[j] = 1 if rng.random() < p_dep else 0
    nA = op.count('A'); nM = op.count('M'); nL = op.count('L'); nS = op.count('S')
    cLU = sum(1 for j in range(1, n) if dep[j] == 1 and op[j - 1] == 'L')
    cMF = sum(1 for j in range(1, n) if dep[j] == 1 and op[j - 1] == 'M')
    cST = sum(1 for j in range(1, n) if op[j - 1] == 'M' and op[j] == 'M')
    return n, nA, nM, nL, nS, cLU, cMF, cST


def gen_held(t):
    """Held-out EXTRAPOLATION probes: much longer consecutive-MUL clusters
    (maxR grows with the case id) than the training table ever shows."""
    rng = random.Random(90001 + t * 15485863)
    maxR_held = 3 + 2 * t
    rows = []
    for _ in range(N_HELD):
        n = rng.randint(N_HELD_LO, N_HELD_HI)
        n_, nA, nM, nL, nS, cLU, cMF, cST = gen_program(rng, n, maxR_held, P_DEP_HELD)
        cyc = true_cycles(n_, cLU, cMF, cST) + rng.gauss(0.0, NOISE_HELD_SIGMA)
        rows.append((n_, nA, nM, nL, nS, cLU, cMF, cST, cyc))
    return rows


# ---------- expression parsing / validation ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
    ast.USub, ast.UAdd,
)


def _validate(tree):
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return "disallowed syntax %s" % type(node).__name__
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS):
                return "disallowed call"
            if node.keywords or len(node.args) != 1:
                return "bad function arity"
        if isinstance(node, ast.Name):
            if node.id in ALLOWED_FUNCS or node.id in ALLOWED_NAMES:
                continue
            return "unknown name %s" % node.id
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                return "non-numeric constant"
            v = float(node.value)
            if v != v or v in (float("inf"), float("-inf")):
                return "non-finite constant"
    return None


def _count_nodes(tree):
    return sum(1 for nd in ast.walk(tree)
               if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant)))


def parse_expr(raw):
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        fail("empty submission")
    text = lines[-1]                       # take the last non-empty line
    if text.upper().startswith("EXPR "):   # optional leading tag
        text = text[5:].strip()
    if not text:
        fail("empty expression")
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    err = _validate(tree)
    if err:
        fail(err)
    nodes = _count_nodes(tree)
    if nodes > MAX_NODES:
        fail("expression too large (%d nodes)" % nodes)
    try:
        code = compile(tree, "<expr>", "eval")
    except Exception:
        fail("compile error")
    return code, nodes


def eval_at(code, n, nA, nM, nL, nS, cLU, cMF, cST):
    env = dict(ALLOWED_FUNCS)
    env["n"] = n; env["nA"] = nA; env["nM"] = nM; env["nL"] = nL; env["nS"] = nS
    env["cLU"] = cLU; env["cMF"] = cMF; env["cST"] = cST
    try:
        p = eval(code, {"__builtins__": {}}, env)
    except Exception:
        return None
    if isinstance(p, bool) or not isinstance(p, (int, float)):
        return None                        # rejects complex results from neg**frac
    p = float(p)
    if p != p or p in (float("inf"), float("-inf")):
        return None
    return p


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            lines = fh.read().splitlines()
        header = lines[0].split()
        n_train = int(header[0]); t = int(header[1])
        train_rows = []
        for ln in lines[1:1 + n_train]:
            parts = ln.split()
            vals = [int(x) for x in parts[:8]] + [float(parts[8])]
            train_rows.append(vals)
        if len(train_rows) != n_train:
            raise ValueError("short train table")
    except Exception:
        fail("bad instance")
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

    code, nodes = parse_expr(text)

    held = gen_held(t)
    ds = []
    for n, nA, nM, nL, nS, cLU, cMF, cST, cyc in held:
        p = eval_at(code, n, nA, nM, nL, nS, cLU, cMF, cST)
        if p is None:
            fail("non-finite / invalid prediction")
        d = abs(p - cyc) / (abs(p) + abs(cyc) + 1e-9)
        ds.append(min(CAP, d))
    metric = sum(ds) / len(ds)

    # baseline: constant predictor = mean(train cycles), read from the given instance
    train_mean = sum(r[8] for r in train_rows) / len(train_rows)
    bd = [min(CAP, abs(train_mean - cyc) / (abs(train_mean) + abs(cyc) + 1e-9)) for *_, cyc in held]
    Bmetric = sum(bd) / len(bd)

    B = Bmetric * (1.0 + LAMBDA * 1)
    O = metric * (1.0 + LAMBDA * nodes)
    sc = min(1000.0, 100.0 * B / max(1e-12, O))
    print("metric=%.6f baseline=%.6f nodes=%d  Ratio: %.6f"
          % (metric, Bmetric, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
