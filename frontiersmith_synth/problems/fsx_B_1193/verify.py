#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the "shape-memory strip" hysteresis-loop-predict task.

- Reads the test id from <in> (header), then regenerates the hidden strip law
  (centerline poly + branch gap, HELD ONLY here and in gen.py -- never
  imported between them) and a HELD-OUT, MORE-AGITATED elongation path
  (more reversals, a different sampling rate, often a wider range than
  training) entirely from that id.
- Parses the participant's closed-form expression over two variables:
      x  -- the current elongation
      b  -- the current branch state (+1 loading / -1 unloading)
  `b` is NOT computed by the participant: the grader derives it from the
  held-out x-sequence via the FIXED, stated rule (b[0]=+1; then the sign of
  x[i]-x[i-1], holding on an exact tie) and substitutes it at each step --
  exactly the "internal state with a specified update rule" the statement
  promises.
- Rolls the expression forward over the held-out path, scores held-out MSE
  against a constant-predictor baseline the grader builds itself, using a
  SELF-NORMALISED scale S = MSE_baseline / K (K fixed) so the mapping from
  "how many x better than baseline" to score is identical across test ids
  regardless of each id's own noise/amplitude scale:
      F = 1 / (1 + MSE / S)          B = 1 / (1 + K)   (constant, = 1/(1+K))
      Ratio = min(1000, 100*F/B) / 1000
  A constant reproduces the baseline exactly (score = 1/(1+K) -> 0.1). A
  memoryless fit of x alone can never separate the two branches (same x
  visited by both loading and unloading with different y) so it plateaus
  well below strong. Recovering the branch-modulated law drives MSE toward
  the sensor-noise floor, which caps the achievable score at
  100*(1+K)/1000 < 1 -- noise keeps even an exact-law fit off the ceiling,
  leaving headroom.
"""
import sys, math, ast, random

SCALE_K = 7.5   # asymptotic cap (MSE->0) = 100*(1+SCALE_K)/1000 = 0.85
MAX_NODES = 50
MAX_OUT_BYTES = 20000
MAX_POW = 4


def fail(reason):
    print("MSE=nan  Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden strip law (duplicated from gen.py, never imported) ----------
def hidden_params(t):
    rng = random.Random(900001 + 7919 * t)
    k0 = rng.uniform(-0.25, 0.25)
    k1 = rng.uniform(0.60, 1.30)
    k2 = rng.uniform(-0.35, 0.35)
    k3 = rng.uniform(-0.45, 0.45)
    c0 = rng.uniform(0.18, 0.32)
    c1 = rng.uniform(0.05, 0.20)
    sigma = 0.010 + 0.0012 * (t - 1)
    return k0, k1, k2, k3, c0, c1, sigma


def centerline(x, k0, k1, k2, k3):
    return k0 + k1 * x + k2 * x * x + k3 * x * x * x


def gap(x, c0, c1):
    return c0 + c1 * x * x


def branch_states(xs):
    b = [1]
    for i in range(1, len(xs)):
        if xs[i] > xs[i - 1]:
            b.append(1)
        elif xs[i] < xs[i - 1]:
            b.append(-1)
        else:
            b.append(b[-1])
    return b


def true_series(xs, params, noise_seed):
    k0, k1, k2, k3, c0, c1, sigma = params
    b = branch_states(xs)
    rng = random.Random(noise_seed)
    ys = []
    for x, bi in zip(xs, b):
        y = centerline(x, k0, k1, k2, k3) + bi * gap(x, c0, c1) + rng.gauss(0.0, sigma)
        ys.append(y)
    return ys, b


def make_path(rng, n_segments, pts_per_seg, x_lo, x_hi, x_start):
    xs = [x_start]
    cur = x_start
    going_up = True
    span = x_hi - x_lo
    for _ in range(n_segments):
        if going_up:
            target = rng.uniform(x_hi - 0.20 * span, x_hi)
        else:
            target = rng.uniform(x_lo, x_lo + 0.20 * span)
        if going_up and target <= cur:
            target = cur + rng.uniform(0.15, 0.4)
        if (not going_up) and target >= cur:
            target = cur - rng.uniform(0.15, 0.4)
        target = min(x_hi, max(x_lo, target))
        m = pts_per_seg()
        fracs = sorted(rng.uniform(0.0, 1.0) for _ in range(m))
        fracs = [(0.06 + 0.88 * f) for f in fracs]
        for f in fracs:
            v = cur + (target - cur) * f
            if not xs or abs(v - xs[-1]) > 1e-4:
                xs.append(round(v, 6))
        xs.append(round(target, 6))
        cur = target
        going_up = not going_up
    out = [xs[0]]
    for v in xs[1:]:
        if abs(v - out[-1]) > 1e-4:
            out.append(v)
    return out


def heldout_path(t):
    """HELD-OUT path: regenerated only here.  More reversals, a DIFFERENT
    sampling density (rate-independence stress test), often a wider
    elongation range than the training path for the same test id."""
    rng = random.Random(20261 + t * 15485863)
    n_seg = rng.randint(3, 6)
    if t <= 4:
        lo, hi = -0.65, 0.65
    elif t <= 7:
        lo, hi = -0.88, 0.88
    else:
        lo, hi = -1.0, 1.0
    dens_mode = rng.choice(["sparse", "dense", "mixed"])

    def pts():
        if dens_mode == "sparse":
            return rng.randint(3, 7)
        if dens_mode == "dense":
            return rng.randint(16, 28)
        return rng.randint(4, 20)

    xs = make_path(rng, n_seg, pts, lo, hi, x_start=lo * 0.5)
    return xs


# ---------- expression parsing / validation ----------
def _sexp(v):
    v = max(-40.0, min(40.0, v))
    return math.exp(v)


ALLOWED_FUNCS = {
    "sin": math.sin,
    "cos": math.cos,
    "tanh": math.tanh,
    "exp": _sexp,
    "absv": abs,
}
ALLOWED_NAMES = {"x", "b"}
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
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
            if node.id not in ALLOWED_NAMES and node.id not in ALLOWED_FUNCS:
                return "unknown name %s" % node.id
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if not isinstance(node.right, ast.Constant) or isinstance(node.right.value, bool):
                return "exponent must be a numeric constant"
            ev = node.right.value
            if not isinstance(ev, (int, float)) or abs(ev) > MAX_POW:
                return "exponent out of range"
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
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if not lines:
        fail("empty output")
    text = lines[0].strip()
    if len(lines) > 1:
        # allow, but ignore, trailing blank/comment noise; reject a second
        # real statement (this format takes exactly one expression)
        for extra in lines[1:]:
            if extra.strip() and not extra.strip().startswith("#"):
                fail("more than one statement")
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    err = _validate(tree)
    if err:
        fail(err)
    nodes = _count_nodes(tree)
    if nodes > MAX_NODES:
        fail("program too large (%d nodes)" % nodes)
    if nodes == 0:
        fail("empty expression")
    try:
        code = compile(tree, "<expr>", "eval")
    except Exception:
        fail("compile error")
    return code, nodes


def evaluate(code, xs, bs):
    glob = {"__builtins__": {}}
    preds = []
    for xv, bv in zip(xs, bs):
        env = dict(ALLOWED_FUNCS)
        env["x"] = xv
        env["b"] = bv
        try:
            p = eval(code, glob, env)
        except Exception:
            fail("evaluation error")
        if not isinstance(p, (int, float)) or isinstance(p, bool):
            fail("non-numeric result")
        p = float(p)
        if p != p or p in (float("inf"), float("-inf")):
            fail("non-finite result")
        if abs(p) > 1e6:
            fail("result out of range")
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

    code, nodes = parse_expr(text)

    params = hidden_params(t)
    xs = heldout_path(t)
    ys, bs = true_series(xs, params, noise_seed=8888 + t * 97)

    preds = evaluate(code, xs, bs)

    n = len(ys)
    mse = sum((p - yv) ** 2 for p, yv in zip(preds, ys)) / n
    mean_y = sum(ys) / n
    mse_base = sum((mean_y - yv) ** 2 for yv in ys) / n

    scale = max(1e-9, mse_base) / SCALE_K
    F = 1.0 / (1.0 + mse / scale)
    B = 1.0 / (1.0 + SCALE_K)
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("n_heldout=%d nodes=%d MSE=%.6f baseline_MSE=%.6f  Ratio: %.6f"
          % (n, nodes, mse, mse_base, sc / 1000.0))


if __name__ == "__main__":
    main()
