#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the storm-season harbour congestion task.

- Reads the test id and the calm-season train rows from <in>; regenerates the
  hidden berth-priority law (constants a0,a1,a2 and the priority permutation)
  plus the STORM-SEASON held-out rows (rho in [0.85, 0.97], fresh mixes,
  multiplicative noise) entirely from the test id.  The hidden law lives ONLY
  here (and in the generator's hidden half).
- Parses the participant's K predictor expressions "Wc = <expr>" over the
  variables rho, m1..mK, r1..rK, h and the functions sig/tanh/relu/absv/
  exp/log/sqrt.  Strict validation: schema, known names, finite constants,
  node limits; every prediction must be finite and strictly positive.
- Score (minimisation):
      e   = min(1, |pred - w| / w)   per held-out row & class
      F   = mean(e) * (1 + LAMBDA * nodes)
      B   = mean(e_baseline) * (1 + LAMBDA * K)   # per-class calm-season means
      Ratio = min(1000, 100*B/max(F,1e-9)) / 1000
  A constant-per-class predictor reproduces B (~0.1).  A smooth polynomial in
  rho is excellent on calm rows but stays finite at the capacity pole, so it
  collapses on storm rows; committing to the pole + recovering the priority
  order extrapolates -- but the unmodelled numerator structure and held-out
  noise keep even that well below the ceiling.
"""
import sys, math, ast, random, re

LAMBDA = 0.002
N_HELD = 48
MAX_NODES_EXPR = 120
MAX_NODES_TOTAL = 400
MAX_OUT_BYTES = 200000


def _clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


ALLOWED_FUNCS = {
    "sig": lambda x: 1.0 / (1.0 + math.exp(-_clamp(x, -60.0, 60.0))),
    "tanh": math.tanh,
    "relu": lambda x: x if x > 0 else 0.0,
    "absv": abs,
    "exp": lambda x: math.exp(_clamp(x, -60.0, 60.0)),
    "log": lambda x: math.log(max(1e-12, x)),
    "sqrt": lambda x: math.sqrt(max(0.0, x)),
}

_LINE_RE = re.compile(r"^W(\d+)\s*=\s*(.+)$")
_NAME_RE = re.compile(r"^([mr])(\d+)$")


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden law (identical to gen.py) ----------
def params(t):
    rng = random.Random(917331 + t * 7919)
    K = 3 + (t - 1) % 3
    a0 = rng.uniform(0.10, 0.20)
    a1 = rng.uniform(0.90, 1.50)
    a2 = rng.uniform(0.50, 0.90)
    perm = list(range(K))
    rng.shuffle(perm)
    skew = 1 if t >= 6 else 0
    return K, a0, a1, a2, perm, skew


def draw_mix(rng, K, skew, storm=False):
    lo = 0.15 if skew else 0.35
    pw = 1.7 if storm else 1.0            # storm season: sparser, concentrated mixes
    u = [rng.uniform(lo, 1.0) ** pw for _ in range(K)]
    s = sum(u)
    return [x / s for x in u]


def true_waits(rho, mix, K, a0, a1, a2, perm):
    r = [rho * m for m in mix]
    h = sum(m * m for m in mix)
    P = [0.0] * K
    cum = 0.0
    for rank in range(K):
        c = perm[rank]
        P[c] = cum
        cum += r[c]
    N = a0 * rho * (1.0 + a1 * h)
    w = []
    for c in range(K):
        X = P[c] + r[c]
        w.append(N * (1.0 + a2 * r[c]) / ((1.0 - P[c]) * (1.0 - X)))
    return w


def heldout_rows(t, K, a0, a1, a2, perm, skew):
    """Storm-season grading rows; regenerated here only, from the test id."""
    rng = random.Random(202607 + t * 15485863)
    sig = 0.055 + 0.004 * t
    rows = []
    for _ in range(N_HELD):
        rho = rng.uniform(0.85, 0.97)
        mix = draw_mix(rng, K, skew, storm=True)
        w = true_waits(rho, mix, K, a0, a1, a2, perm)
        w = [max(1e-6, wc * (1.0 + rng.gauss(0.0, sig))) for wc in w]
        rows.append((rho, mix, w))
    return rows


# ---------- expression parsing / validation ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
)


def _validate_ast(tree, K):
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
            if nm in ALLOWED_FUNCS or nm in ("rho", "h"):
                continue
            mm = _NAME_RE.match(nm)
            if mm and 1 <= int(mm.group(2)) <= K:
                continue
            fail("unknown name %s" % nm)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                fail("non-numeric constant")
            v = float(node.value)
            if v != v or v in (float("inf"), float("-inf")):
                fail("non-finite constant")


def _count_nodes(tree):
    return sum(1 for nd in ast.walk(tree)
               if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant)))


def parse_program(raw, K):
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        fail("empty program")
    exprs = {}
    for ln in lines:
        m = _LINE_RE.match(ln)
        if not m:
            fail("bad line '%s'" % ln[:40])
        k = int(m.group(1))
        if k < 1 or k > K:
            fail("label out of range")
        if k in exprs:
            fail("duplicate label W%d" % k)
        text = m.group(2)
        if "|" in text:
            fail("stray '|'")
        try:
            tree = ast.parse(text, mode="eval")
        except Exception:
            fail("parse error")
        _validate_ast(tree, K)
        exprs[k] = tree
    if len(exprs) != K:
        fail("need exactly K=%d expressions" % K)
    nodes = 0
    codes = {}
    for k in range(1, K + 1):
        n = _count_nodes(exprs[k])
        if n > MAX_NODES_EXPR:
            fail("expression W%d too large" % k)
        nodes += n
        try:
            codes[k] = compile(exprs[k], "<dsl>", "eval")
        except Exception:
            fail("compile error")
    if nodes > MAX_NODES_TOTAL:
        fail("program too large")
    return codes, nodes


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    # ---- read instance: header + calm-season train rows ----
    try:
        with open(inf) as fh:
            toks = fh.read().split()
        n, K, t = int(toks[0]), int(toks[1]), int(toks[2])
        assert n >= 1 and 1 <= K <= 12 and 1 <= t <= 100000
        vals = toks[3:]
        assert len(vals) == n * (1 + 2 * K)
        wsum = [0.0] * K
        idx = 0
        for _ in range(n):
            idx += 1 + K                    # skip rho and mix
            for c in range(K):
                wsum[c] += float(vals[idx])
                idx += 1
        wbar = [s / n for s in wsum]
    except Exception:
        fail("bad instance file")

    # ---- read participant program ----
    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace")
    codes, nodes = parse_program(text, K)

    # ---- regenerate hidden law + storm-season rows ----
    Kh, a0, a1, a2, perm, skew = params(t)
    if Kh != K:
        fail("instance/grader mismatch")
    rows = heldout_rows(t, K, a0, a1, a2, perm, skew)

    # ---- evaluate ----
    glob = {"__builtins__": {}}
    tot = 0.0
    cnt = 0
    for rho, mix, w in rows:
        env = dict(ALLOWED_FUNCS)
        env["rho"] = rho
        env["h"] = sum(m * m for m in mix)
        for c in range(K):
            env["m%d" % (c + 1)] = mix[c]
            env["r%d" % (c + 1)] = rho * mix[c]
        for c in range(K):
            try:
                p = eval(codes[c + 1], glob, env)
            except Exception:
                fail("evaluation error")
            if not isinstance(p, (int, float)) or isinstance(p, bool):
                fail("non-numeric prediction")
            p = float(p)
            if p != p or p in (float("inf"), float("-inf")):
                fail("non-finite prediction")
            if p <= 0.0:
                fail("non-positive prediction")
            e = abs(p - w[c]) / w[c]
            tot += min(1.0, e)
            cnt += 1
    E = tot / cnt

    # ---- internal baseline: per-class calm-season mean wait ----
    btot = 0.0
    for rho, mix, w in rows:
        for c in range(K):
            e = abs(wbar[c] - w[c]) / w[c]
            btot += min(1.0, e)
    E_base = btot / cnt

    F = E * (1.0 + LAMBDA * nodes)
    B = E_base * (1.0 + LAMBDA * K)
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("held_err=%.6f base_err=%.6f nodes=%d  Ratio: %.6f"
          % (E, E_base, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
