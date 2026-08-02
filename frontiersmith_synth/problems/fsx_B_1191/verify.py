#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the PV-inverter-clipping-forecast task.

- Reads the test id + nameplate N from <in> header.
- Regenerates the hidden per-site physics (array efficiency `eff`, inverter
  clip fraction `rho`) and the HELD-OUT FAST SUMMER trace entirely from the
  test id (identical `site_params` to gen.py -- the hidden law lives ONLY
  here and in gen.py, never in a shared importable module).
- Parses the participant's closed-form power expression: a single arithmetic
  expression over `G` (irradiance), `T` (temperature), `N` (the site's
  nameplate constant), `+ - * /`, parentheses, numeric constants, and the
  functions `min`, `max` (2-arg) and `absv` (1-arg).
- Evaluates the expression row-by-row on the held-out summer trace, scores
  held-out MSE with a small node-count parsimony penalty (minimisation of
  error -> converted to a MAXIMISE ratio):
      F = heldout_MSE * (1 + LAMBDA * nodes)
      B = baseline_MSE * (1 + LAMBDA * nodes_of_baseline)   # baseline = 0.4*N
      Ratio = min(1000, 100*B/F) / 1000
  A flat 0.4*N reproduces the baseline (~0.1). An unbounded fit of the
  visible (never-clipped) winter branch improves on that but is dragged back
  down by the many held-out rows where the true power is flat at the clip
  and its own prediction keeps climbing. A model that also recognises a hard
  ceiling must exist (using N and the stated clip-fraction RANGE) closes
  most of that gap -- but sensor/microclimate noise plus imperfect knowledge
  of the exact clip fraction keep even a good model well below 1.0, leaving
  headroom.
"""
import sys, math, ast

LAMBDA = 0.01
NH = 300               # held-out row count
MAX_OUT_BYTES = 20000
MAX_NODES = 40
BASELINE_FRAC = 0.4     # internal baseline: predict a flat 0.4 * N


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden per-site physics (verbatim copy of gen.py's site_params) ----------
def site_params(t):
    import random
    rng = random.Random(90210 + t * 7919)
    N = rng.uniform(50.0, 150.0)
    eff = rng.uniform(0.92, 1.03)
    rho = rng.uniform(0.58, 0.88)
    return N, eff, rho


SIGMA_FRAC = 0.11


def held_rows(t, n):
    import random
    rng = random.Random(500029 + t * 15485863)   # different stream than train
    N, eff, rho = site_params(t)
    Pcap = rho * N
    rows = []
    for _ in range(n):
        T = rng.uniform(15.0, 38.0)
        G = rng.uniform(0.0, 1150.0)
        factor = 1.0 - 0.004 * (T - 25.0)
        raw = eff * N * (G / 1000.0) * factor
        P = min(raw, Pcap) + rng.gauss(0.0, SIGMA_FRAC * N)
        P = max(0.0, P)
        rows.append((G, T, P))
    return rows, N


# ---------- expression parsing / validation ----------
ALLOWED_FUNCS_1 = {"absv": abs}
ALLOWED_FUNCS_2 = {
    "min": lambda a, b: a if a < b else b,
    "max": lambda a, b: a if a > b else b,
}
_ALLOWED_NAMES = {"G", "T", "N"}
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
)


def _validate_ast(tree):
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return "disallowed syntax %s" % type(node).__name__
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                return "disallowed call target"
            fname = node.func.id
            if node.keywords:
                return "no keyword args allowed"
            if fname in ALLOWED_FUNCS_1:
                if len(node.args) != 1:
                    return "%s takes exactly 1 arg" % fname
            elif fname in ALLOWED_FUNCS_2:
                if len(node.args) != 2:
                    return "%s takes exactly 2 args" % fname
            else:
                return "unknown function %s" % fname
        if isinstance(node, ast.Name):
            if node.id in _ALLOWED_NAMES:
                continue
            if node.id in ALLOWED_FUNCS_1 or node.id in ALLOWED_FUNCS_2:
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


class _SafeDiv(ast.NodeTransformer):
    """No-op placeholder kept for clarity; division-by-zero is caught at eval time."""
    pass


def compile_expr(text):
    text = text.strip()
    if not text:
        fail("empty expression")
    if len(text) > 4000:
        fail("expression too long")
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    err = _validate_ast(tree)
    if err:
        fail(err)
    nodes = _count_nodes(tree)
    if nodes > MAX_NODES:
        fail("program too large (%d nodes)" % nodes)
    try:
        code = compile(tree, "<expr>", "eval")
    except Exception:
        fail("compile error")
    return code, nodes


def eval_expr(code, G, T, N):
    env = {"G": G, "T": T, "N": N}
    env.update(ALLOWED_FUNCS_1)
    env.update(ALLOWED_FUNCS_2)
    try:
        v = eval(code, {"__builtins__": {}}, env)
    except ZeroDivisionError:
        return None
    except Exception:
        return None
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return None
    v = float(v)
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            header = fh.readline().split()
        t = int(header[1])
        N_in = float(header[2])
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
    text = raw.decode("utf-8", "replace").strip()
    # participant may emit a bare expression, optionally prefixed "EXPR "
    if text.upper().startswith("EXPR "):
        text = text[5:].strip()
    if "\n" in text:
        # only the first non-empty line is the expression
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) != 1:
            fail("expected exactly one expression line")
        text = lines[0].strip()

    code, nodes = compile_expr(text)

    held, N = held_rows(t, NH)
    if abs(N - N_in) > 1e-3:
        fail("instance corrupted")  # internal consistency guard, not participant-facing

    se = 0.0
    for G, T, P in held:
        pred = eval_expr(code, G, T, N)
        if pred is None:
            fail("non-finite or invalid value produced during evaluation")
        se += (pred - P) ** 2
    F_mse = se / len(held)

    base_code, base_nodes = compile_expr("%.10f * N" % BASELINE_FRAC)
    se_b = 0.0
    for G, T, P in held:
        pb = eval_expr(base_code, G, T, N)
        se_b += (pb - P) ** 2
    B_mse = se_b / len(held)

    F = F_mse * (1.0 + LAMBDA * nodes)
    B = B_mse * (1.0 + LAMBDA * base_nodes)
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("heldout_MSE=%.6f baseline_MSE=%.6f nodes=%d  Ratio: %.6f"
          % (F_mse, B_mse, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
