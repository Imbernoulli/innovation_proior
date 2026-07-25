#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the collision-rig conserved-quantity task.

- Reads the test id from the <in> header, then regenerates the hidden law
  (shared exponent alpha, restitution e0*exp(-beta*gamma - eta*|w|)) and a
  HELD-OUT grading rig whose masses lie OUTSIDE every training rig's range
  (harder ids: further outside; hard ids also extrapolate damping).  The
  hidden law lives ONLY here and in gen.py; it is never printed.
- Parses the participant predictor, a tiny expression program:

      LET <name> <expr>      (optional, up to 8; evaluated in order)
      V1 <expr>              (required: predicted post-impact velocity of body 1)
      V2 <expr>              (required: predicted post-impact velocity of body 2)

  Expressions are arithmetic (+ - * / **, unary -) over the rig constants
  m1, m2, g, the pre-impact velocities v1, v2, previously defined LET names,
  numeric constants, and the unary functions exp, sqrt, absv, tanh, sig.
- The predictor is evaluated on every held-out collision and scored by MSE
  against the noisy measured post-impact velocities, with a small node-count
  parsimony penalty (minimisation):

      F = heldout_MSE * (1 + LAMBDA * nodes)
      B = baseline_MSE * (1 + LAMBDA * 2)   # baseline = identity "nothing happens"
      Ratio = min(1000, 100*B/max(1e-9,F)) / 1000

  Predicting the pre-impact state unchanged reproduces the baseline (~0.1).
  A pooled black-box curve interpolates inside the training mass range but
  extrapolates badly to the new mass regime.  Recovering the conserved
  quantity's functional form (cross-rig invariant search) generalises; the
  impact-speed dependence of the restitution plus sensor noise keeps even a
  good recovery below the ceiling, leaving headroom.
"""
import sys, math, ast, random, re

LAMBDA = 0.001
NH = 240
K = 5
MAX_NODES = 200
MAX_LETS = 8
MAX_OUT_BYTES = 65536
MAX_CONST = 1e9

ALLOWED_FUNCS = {
    "exp": lambda x: math.exp(max(-60.0, min(60.0, x))),
    "sqrt": math.sqrt,
    "absv": abs,
    "tanh": math.tanh,
    "sig": lambda x: 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x)))),
}
BASE_VARS = {"m1", "m2", "g", "v1", "v2"}
RESERVED = BASE_VARS | set(ALLOWED_FUNCS)
LET_RE = re.compile(r"^[a-z][a-z0-9]{0,7}$")


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden law (identical to gen.py; never printed) ----------
def hidden(t):
    rng = random.Random(917331 + t * 7919)
    if t <= 3:
        alpha = rng.uniform(0.94, 1.06)
    elif t <= 6:
        alpha = rng.choice([rng.uniform(0.84, 0.93), rng.uniform(1.07, 1.16)])
    else:
        alpha = rng.choice([rng.uniform(0.70, 0.82), rng.uniform(1.18, 1.32)])
    e0 = rng.uniform(0.55, 0.80)
    beta = rng.uniform(0.40, 1.00)
    eta = rng.uniform(0.55, 0.95)
    return alpha, e0, beta, eta


def sigma0(t):
    return 0.10 + 0.005 * t


def vscale(t):
    return 1.0 + 0.02 * t


def collide(m1, m2, g, v1, v2, alpha, e0, beta, eta):
    a1 = m1 ** alpha
    a2 = m2 ** alpha
    w = v2 - v1
    e = e0 * math.exp(-beta * g - eta * abs(w))
    P = a1 * v1 + a2 * v2
    v1p = (P + a2 * e * w) / (a1 + a2)
    v2p = (P - a1 * e * w) / (a1 + a2)
    return v1p, v2p


def heldout_rig(t):
    """Held-out grading rig; masses OUTSIDE the training band [1,8]."""
    rng = random.Random(555121 + t * 15485863)
    lo = 9.0 + 1.2 * t
    hi = lo + 2.5 + 0.6 * t
    m1 = rng.uniform(lo, hi)
    m2 = rng.uniform(lo, hi)
    g = rng.uniform(0.05, 0.40) if t <= 5 else rng.uniform(0.42, 0.60)
    dt = rng.choice([0.5, 1.0, 2.0])
    return m1, m2, g, dt


def heldout_rows(t):
    alpha, e0, beta, eta = hidden(t)
    m1, m2, g, dt = heldout_rig(t)
    s = vscale(t)
    rngv = random.Random(31337 + t * 97)
    rngn = random.Random(777 + t * 17)
    sig = sigma0(t) * dt
    rows = []
    for _ in range(NH):
        v1 = s * rngv.uniform(-1.1, 0.4)
        v2 = s * rngv.uniform(-0.4, 1.1)
        y1, y2 = collide(m1, m2, g, v1, v2, alpha, e0, beta, eta)
        y1 += rngn.gauss(0.0, sig)
        y2 += rngn.gauss(0.0, sig)
        rows.append((v1, v2, y1, y2))
    return m1, m2, g, rows


# ---------- DSL parsing / validation ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
    ast.USub, ast.UAdd,
)


def _compile_expr(text, env_names):
    text = text.strip()
    if not text:
        fail("empty expression")
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            fail("disallowed syntax %s" % type(node).__name__)
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS):
                fail("disallowed call")
            if node.keywords or len(node.args) != 1:
                fail("bad function arity")
        if isinstance(node, ast.Name):
            nm = node.id
            if nm in ALLOWED_FUNCS or nm in BASE_VARS or nm in env_names:
                continue
            fail("unknown name %s" % nm)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                fail("non-numeric constant")
            v = float(node.value)
            if v != v or v in (float("inf"), float("-inf")) or abs(v) > MAX_CONST:
                fail("bad constant")
    try:
        code = compile(tree, "<dsl>", "eval")
    except Exception:
        fail("compile error")
    ncount = sum(1 for nd in ast.walk(tree)
                 if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant)))
    return code, ncount


def parse_program(raw):
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        fail("empty program")
    lets = []
    env_names = set()
    codes = {}
    nodes = 0
    for ln in lines:
        head = ln.split(None, 1)
        kw = head[0]
        rest = head[1] if len(head) > 1 else ""
        if kw == "LET":
            if len(lets) >= MAX_LETS:
                fail("too many LETs")
            parts = rest.split(None, 1)
            if len(parts) != 2 or not LET_RE.match(parts[0]) or parts[0] in RESERVED:
                fail("bad LET name")
            name = parts[0]
            if name in env_names:
                fail("LET redefinition")
            code, nc = _compile_expr(parts[1], env_names)
            lets.append((name, code))
            env_names.add(name)
            nodes += nc
        elif kw in ("V1", "V2"):
            if kw in codes:
                fail("multiple %s statements" % kw)
            code, nc = _compile_expr(rest, env_names)
            codes[kw] = code
            nodes += nc
        else:
            fail("unknown statement '%s'" % kw)
    if "V1" not in codes or "V2" not in codes:
        fail("missing V1/V2 statement")
    if nodes > MAX_NODES:
        fail("program too large (%d nodes)" % nodes)
    return lets, codes["V1"], codes["V2"], nodes


# ---------- evaluation ----------
def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) \
        and x == x and x not in (float("inf"), float("-inf"))


def evaluate(lets, c1, c2, m1, m2, g, rows):
    glob = {"__builtins__": {}}
    se = 0.0
    for (v1, v2, y1, y2) in rows:
        env = dict(ALLOWED_FUNCS)
        env.update(m1=m1, m2=m2, g=g, v1=v1, v2=v2)
        try:
            for name, code in lets:
                val = eval(code, glob, env)
                if not _finite(val):
                    fail("non-finite LET value")
                env[name] = float(val)
            p1 = eval(c1, glob, env)
            p2 = eval(c2, glob, env)
        except SystemExit:
            raise
        except Exception:
            fail("evaluation error")
        if not _finite(p1) or not _finite(p2):
            fail("non-finite prediction")
        se += (float(p1) - y1) ** 2 + (float(p2) - y2) ** 2
    return se


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

    lets, c1, c2, nodes = parse_program(text)

    m1, m2, g, rows = heldout_rows(t)
    se = evaluate(lets, c1, c2, m1, m2, g, rows)
    n2 = 2.0 * len(rows)
    F_mse = se / n2
    B_mse = sum((v1 - y1) ** 2 + (v2 - y2) ** 2 for (v1, v2, y1, y2) in rows) / n2

    B = B_mse * (1.0 + LAMBDA * 2)
    F = F_mse * (1.0 + LAMBDA * nodes)
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("heldout_MSE=%.6f baseline_MSE=%.6f nodes=%d  Ratio: %.6f"
          % (F_mse, B_mse, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
