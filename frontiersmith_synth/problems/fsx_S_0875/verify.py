#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for "why a drone fleet slows near itself".

- Reads the test id from <in> header, then regenerates the hidden interference
  law (kernel constants a,c) AND the HELD-OUT swarm rows entirely from that id.
  The ground truth lives ONLY here (and, identically, inside gen.py -- never
  printed to the solver).
- The held-out rows are LARGE swarms (20-40 sensed neighbors) flown roughly 3x
  denser than anything in TRAIN (which only ever showed 3-8 drone fleets) --
  genuine density extrapolation, not interpolation.
- Parses the participant's two-statement program in a tiny arithmetic DSL:
      KERNEL <expr over "dist">      -- the pairwise interference term
      OUT    <expr over "S" and "v"> -- S = sum of KERNEL(dist) over sensed
                                         neighbors; v = commanded speed
  Expressions: + - * / , parens, numeric constants, unary minus, and ONLY the
  named variables above (no function calls).
- For every held-out row the grader evaluates KERNEL on each listed neighbor
  distance, sums to S, evaluates OUT(S, v), and compares to the true realized
  speed. Score = held-out MSE turned into a bounded "goodness" and compared to
  a trivial baseline (predict realized == commanded, i.e. ignore interference).
"""
import sys, math, ast, random

MAX_NODES = 50
MAX_OUT_BYTES = 20000
N_HELD = 26


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden law (identical to gen.py) ----------
def law_params(t):
    rng = random.Random(9130007 + t * 7919)
    a = rng.uniform(0.85, 1.55)
    c = rng.uniform(0.06, 0.24)
    return a, c


def true_kernel(d, a, c):
    return a / (d * d + c)


def true_outer(S):
    return 1.0 / (1.0 + S)


def held_out_rows(t, a, c):
    """Regenerate the held-out swarm rows: 20-40 neighbors, ~3x denser spacing
    than anything TRAIN ever showed (TRAIN distances were drawn from
    [0.45, 3.2]; held-out distances are drawn from [0.15, 1.05])."""
    rng = random.Random(70310501 + t * 15485863)
    sigma = 0.028 + 0.006 * (t - 1)
    dmin, dmax = 0.15, 1.05
    rows = []
    for _ in range(N_HELD):
        k = rng.randint(20, 40)
        v = rng.uniform(2.0, 6.0)
        dists = [rng.uniform(dmin, dmax) for _ in range(k)]
        S = sum(true_kernel(d, a, c) for d in dists)
        y = v * true_outer(S) + rng.gauss(0.0, sigma)
        y = max(0.0, y)
        rows.append((v, dists, y))
    return rows


# ---------- DSL parsing / validation ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
)


def _validate_ast(tree, allowed_names):
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return "disallowed syntax %s" % type(node).__name__
        if isinstance(node, ast.Name):
            if node.id not in allowed_names:
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
               if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Name, ast.Constant)))


def _compile_expr(text, allowed_names, label):
    text = text.strip()
    if not text:
        fail("empty %s expression" % label)
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("%s parse error" % label)
    err = _validate_ast(tree, allowed_names)
    if err:
        fail("%s: %s" % (label, err))
    n = _count_nodes(tree)
    if n > MAX_NODES:
        fail("%s expression too large (%d nodes)" % (label, n))
    try:
        code = compile(tree, "<dsl>", "eval")
    except Exception:
        fail("%s compile error" % label)
    return code, n


def parse_program(raw):
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        fail("empty program")
    kernel_code = out_code = None
    total_nodes = 0
    for ln in lines:
        head = ln.split(None, 1)
        kw = head[0].upper()
        rest = head[1] if len(head) > 1 else ""
        if kw == "KERNEL":
            if kernel_code is not None:
                fail("multiple KERNEL statements")
            kernel_code, n1 = _compile_expr(rest, {"dist"}, "KERNEL")
            total_nodes += n1
        elif kw == "OUT":
            if out_code is not None:
                fail("multiple OUT statements")
            out_code, n2 = _compile_expr(rest, {"S", "v"}, "OUT")
            total_nodes += n2
        else:
            fail("unknown statement '%s'" % kw)
    if kernel_code is None:
        fail("missing KERNEL statement")
    if out_code is None:
        fail("missing OUT statement")
    if total_nodes > MAX_NODES:
        fail("program too large (%d nodes)" % total_nodes)
    return kernel_code, out_code


_SAFE_GLOB = {"__builtins__": {}}


def eval_kernel(code, d):
    try:
        r = eval(code, _SAFE_GLOB, {"dist": d})
    except ZeroDivisionError:
        fail("division by zero in KERNEL")
    except Exception:
        fail("evaluation error in KERNEL")
    if not isinstance(r, (int, float)) or isinstance(r, bool):
        fail("non-numeric KERNEL result")
    r = float(r)
    if r != r or r in (float("inf"), float("-inf")):
        fail("non-finite KERNEL result")
    return r


def eval_out(code, S, v):
    try:
        r = eval(code, _SAFE_GLOB, {"S": S, "v": v})
    except ZeroDivisionError:
        fail("division by zero in OUT")
    except Exception:
        fail("evaluation error in OUT")
    if not isinstance(r, (int, float)) or isinstance(r, bool):
        fail("non-numeric OUT result")
    r = float(r)
    if r != r or r in (float("inf"), float("-inf")):
        fail("non-finite OUT result")
    return r


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

    kernel_code, out_code = parse_program(text)

    a, c = law_params(t)
    rows = held_out_rows(t, a, c)

    se = 0.0
    for v, dists, y_true in rows:
        S = 0.0
        for d in dists:
            S += eval_kernel(kernel_code, d)
        pred = eval_out(out_code, S, v)
        se += (pred - y_true) ** 2
    mse = se / len(rows)

    # internal baseline: ignore interference entirely (predict realized == commanded)
    base_se = sum((v - y_true) ** 2 for v, _, y_true in rows)
    base_mse = base_se / len(rows)

    # Log-space goodness: MSE spans many orders of magnitude between a naive
    # baseline (dominated by commanded-speed variance, since dense held-out
    # swarms are heavily slowed) and a recovered mechanistic fit (down near
    # the sensor-noise floor). A raw reciprocal-of-MSE ratio saturates the
    # score instantly at that dynamic range, so goodness is measured in
    # log10(MSE) space instead: each order-of-magnitude cut in error is worth
    # a fixed amount of score, and B is fixed at exactly 1.0 by construction
    # (the baseline's own MSE always maps back to F==B==1). EPS bounds the
    # maximum achievable goodness, capping the ceiling.
    EPS = 1e-5
    CAP = 900.0
    L_mse = -math.log10(mse + EPS)
    L_base = -math.log10(base_mse + EPS)
    F = 1.0 + (L_mse - L_base)
    B = 1.0

    sc = min(CAP, 100.0 * F / max(1e-9, B))
    sc = max(0.0, sc)
    print("heldout_MSE=%.6f baseline_MSE=%.6f  Ratio: %.6f" % (mse, base_mse, sc / 1000.0))


if __name__ == "__main__":
    main()
