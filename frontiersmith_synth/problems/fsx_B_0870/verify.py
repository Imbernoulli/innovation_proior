#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the percolation critical-scaling-recovery task.

- Reads the test id (and the training census) from <in>.
- Regenerates the hidden ecosystem (p_c, beta, amplitude A; the crossover width W
  is public, already in the header) entirely from the test id -- IDENTICAL to
  gen.py.  The hidden law lives ONLY here (and in gen.py's private functions).
- Regenerates a HELD-OUT set of bond probabilities AT and ABOVE the transition
  (genuine extrapolation -- disjoint from the training band, which stops strictly
  below the transition) with its own noise, never printed by gen.py.
- Parses the participant's closed-form law S(p) -- an expression over the single
  variable `p`, numeric constants, operators + - * / and unary +/-, and the
  functions absv(a), minv(a,b), maxv(a,b), powv(a,b) [a clamped to >=0 before the
  power, so the domain is always defined].
- Scores by held-out MSE (participant law vs NOISY held-out targets) with a small
  node-count parsimony penalty (minimisation):
      F = heldout_MSE * (1 + LAMBDA * nodes)
      B = baseline_MSE * (1 + LAMBDA * 1)     # baseline: constant = mean(train S)
      Ratio = min(1000, 100*B/F) / 1000
  A flat/smooth fit through the sub-critical census reproduces roughly B; genuine
  recovery of the (p-p_c)^beta singularity drives held-out error down -- but
  substantial held-out measurement noise (irreducible) keeps even a very good law
  well below the ceiling, leaving headroom.
"""
import sys, math, ast, random, re

PC_BASE = [0.30, 0.34, 0.38, 0.42, 0.46, 0.50, 0.55, 0.60, 0.65, 0.70]
BETA_BASE = [1.00, 1.20, 0.90, 1.40, 1.10, 1.60, 1.30, 0.85, 1.50, 1.05]
W_BASE = [0.015, 0.012, 0.020, 0.010, 0.018, 0.008, 0.016, 0.022, 0.009, 0.014]

HO_ZS = [-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
HO_ABS_FLOOR = 0.02
HO_REL = 0.75

LAMBDA = 0.003
MAX_NODES = 200
MAX_OUT_BYTES = 200000
CLAMP = 1.0e4

ALLOWED_FUNCS_ARITY = {"absv": 1, "minv": 2, "maxv": 2, "powv": 2}


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden ecosystem (IDENTICAL to gen.py) ----------
def hidden_params(t):
    rng = random.Random(870000 + t * 7919)
    pc = PC_BASE[t - 1] + rng.uniform(-0.01, 0.01)
    beta = BETA_BASE[t - 1] + rng.uniform(-0.05, 0.05)
    w = W_BASE[t - 1] * rng.uniform(0.9, 1.1)
    far_target = rng.uniform(0.55, 0.80)
    p_far = 0.95
    amp = far_target / ((p_far - pc) ** beta)
    return pc, beta, w, amp


def gfun(z):
    return 0.5 * (z + math.sqrt(z * z + 4.0))


def S_true(p, pc, beta, w, amp):
    z = (p - pc) / w
    val = amp * (w * gfun(z)) ** beta
    return max(0.0, min(1.0, val))


def heldout(t, pc, beta, w, amp):
    zcap = (0.95 - pc) / w
    zs = sorted(set(min(z, zcap) for z in HO_ZS))
    rng = random.Random(5550 + t * 29)
    clean, noisy = [], []
    for z in zs:
        p = pc + z * w
        strue = S_true(p, pc, beta, w, amp)
        sig = max(HO_ABS_FLOOR, HO_REL * abs(strue))
        clean.append((p, strue))
        noisy.append(strue + rng.gauss(0.0, sig))
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
            if not (isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS_ARITY):
                return None, "disallowed call"
            if node.keywords:
                return None, "keyword args not allowed"
            need = ALLOWED_FUNCS_ARITY[node.func.id]
            if len(node.args) != need:
                return None, "%s takes %d arg(s)" % (node.func.id, need)
        if isinstance(node, ast.Name):
            nm = node.id
            if nm in ALLOWED_FUNCS_ARITY:
                continue
            if nm != "p":
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


def _safe_powv(a, b):
    base = a if a > 0.0 else 0.0
    try:
        r = base ** b
    except (OverflowError, ValueError, ZeroDivisionError, ArithmeticError):
        # covers a zero base raised to a negative/non-integer exponent, which is
        # mathematically undefined -- treated as a domain violation like any
        # other non-finite result (caught by the isfinite check in eval_law).
        return float("inf")
    return r


def parse_law(raw):
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        fail("empty output")
    text = lines[-1]
    low = text.lower()
    for pre in ("s(p)", "s_pred", "out", "y"):
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
    return code, nodes


def eval_law(code, p):
    env = {
        "p": float(p),
        "absv": abs,
        "minv": min,
        "maxv": max,
        "powv": _safe_powv,
    }
    glob = {"__builtins__": {}}
    try:
        v = eval(code, glob, env)
    except Exception:
        fail("evaluation error")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        fail("non-numeric law result")
    v = float(v)
    if v != v or v in (float("inf"), float("-inf")):
        fail("non-finite law result")
    if v > CLAMP:
        v = CLAMP
    elif v < -CLAMP:
        v = -CLAMP
    return v


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            lines = fh.read().split("\n")
        header = lines[0].split()
        t, n = int(header[0]), int(header[1])
        train_rows = []
        for i in range(1, n + 1):
            parts = lines[i].split()
            train_rows.append((float(parts[0]), float(parts[1])))
    except Exception:
        fail("bad instance")
    if t < 1 or t > 100000 or n <= 0:
        fail("bad test id / row count")

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace")

    code, nodes = parse_law(text)

    pc, beta, w, amp = hidden_params(t)
    clean, noisy = heldout(t, pc, beta, w, amp)

    preds = [eval_law(code, p) for p, _ in clean]
    F_mse = sum((pr - ny) ** 2 for pr, ny in zip(preds, noisy)) / len(noisy)

    mean_train = sum(s for _, s in train_rows) / len(train_rows)
    B_mse = sum((mean_train - ny) ** 2 for ny in noisy) / len(noisy)

    F = F_mse * (1.0 + LAMBDA * nodes)
    B = B_mse * (1.0 + LAMBDA * 1)
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("heldout_MSE=%.6f baseline_MSE=%.6f nodes=%d  Ratio: %.6f"
          % (F_mse, B_mse, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
