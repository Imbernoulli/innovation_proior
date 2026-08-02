#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for impulse-response-deconvolve.

- Reads the test id `t` from <in>'s header, then regenerates the hidden
  instrument (kernel half-width w, noise sigma -- identical formulas to
  gen.py) and a FRESH, held-out, UNPAIRED trace (new spike train, same
  kernel + comparable noise) entirely from `t`.  The hidden w, sigma and the
  held-out spikes/trace are never written to <in>; they live only here (and
  identically in gen.py for the calibration side).
- Parses the participant's single-line closed-form filter expression over
  window taps ym5..ym1, y0, yp1..yp5 (offsets -5..+5 around a query index),
  the constants, +-*/, and unary funcs sig/step/relu/tanh/absv.
- Rolls the SAME expression, unchanged, over every interior query position
  of the held-out trace (no peeking at the held-out truth), producing a
  predicted spike train x_hat.
- Scores by held-out MSE against the true held-out spikes, with a small
  node-count parsimony tax (minimisation, then normalised to a maximisation
  ratio):
      F = MSE * (1 + LAMBDA * nodes)
      B = MSE_of_all_zero * (1 + LAMBDA * 1)      # internal baseline
      Ratio = min(1000, 100 * B / F) / 1000
  Predicting all-zero reproduces B (Ratio ~ 0.1).  A filter that recovers
  more true signal than it injects noise raises the score; a filter that
  over-resolves past where the kernel actually carries information injects
  MORE noise than it recovers and falls back toward (or below) baseline.
"""
import sys, math, ast, random

R = 7
W_MAX = 5
NHELD = 260
MAX_NODES = 100
MAX_OUT_BYTES = 200000
LAMBDA = 0.0015

ALLOWED_FUNCS = {
    "sig": lambda x: 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x)))),
    "step": lambda x: 1.0 if x > 0 else 0.0,
    "relu": lambda x: x if x > 0 else 0.0,
    "tanh": math.tanh,
    "absv": abs,
}
TAP_NAMES = {}
for _j in range(-R, R + 1):
    if _j < 0:
        TAP_NAMES["ym%d" % (-_j)] = _j
    elif _j == 0:
        TAP_NAMES["y0"] = 0
    else:
        TAP_NAMES["yp%d" % _j] = _j


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden instrument (identical ladder to gen.py) ----------
_TABLE = {
    1:  dict(Ncal=500, w=5, sigma=0.012),
    2:  dict(Ncal=420, w=4, sigma=0.017),
    3:  dict(Ncal=28,  w=2, sigma=0.055),
    4:  dict(Ncal=380, w=3, sigma=0.017),
    5:  dict(Ncal=28,  w=1, sigma=0.060),
    6:  dict(Ncal=330, w=5, sigma=0.024),
    7:  dict(Ncal=290, w=2, sigma=0.026),
    8:  dict(Ncal=270, w=4, sigma=0.026),
    9:  dict(Ncal=230, w=3, sigma=0.028),
    10: dict(Ncal=28,  w=1, sigma=0.065),
}


def plan_for(t):
    base = ((t - 1) % 10) + 1
    growth = (t - 1) // 10
    d = dict(_TABLE[base])
    d["sigma"] = d["sigma"] * (1.0 + 0.15 * growth)
    d["Ncal"] = max(18, d["Ncal"] - 5 * growth)
    return d


def kernel(w):
    z = float((w + 1) ** 2)
    return {j: (w + 1 - abs(j)) / z for j in range(-w, w + 1)}


def make_spikes(rng, n, w, target_k):
    min_gap = 2 * w + 3
    positions = []
    tries = 0
    while len(positions) < target_k and tries < 30000:
        tries += 1
        p = rng.randint(2, n - 3)
        if all(abs(p - q) >= min_gap for q in positions):
            positions.append(p)
    x = [0.0] * n
    for p in positions:
        x[p] = rng.uniform(1.0, 3.0)
    return x


def convolve(x, w, sigma, rng):
    n = len(x)
    h = kernel(w)
    y = [0.0] * n
    for nn in range(n):
        s = 0.0
        for j in range(-w, w + 1):
            k = nn - j
            if 0 <= k < n:
                s += h[j] * x[k]
        y[nn] = s + rng.gauss(0.0, sigma)
    return y


def held_rngs(t):
    """RNG streams for the HELD-OUT side only -- disjoint seeds from gen.py's calibration
    streams, so the held-out trace is a genuinely fresh draw, never printed anywhere."""
    r_spk = random.Random(33_000_233 + t * 6151)
    r_noi = random.Random(44_000_501 + t * 5003)
    return r_spk, r_noi


# ---------- expression parsing / validation ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
)


def parse_expr(raw):
    text = raw.strip()
    if not text:
        fail("empty output")
    if "\n" in text.strip("\n"):
        # allow a single trailing blank; multiple non-blank lines are rejected
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) != 1:
            fail("output must be a single expression line")
        text = lines[0].strip()
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
            if nm in ALLOWED_FUNCS:
                continue
            if nm not in TAP_NAMES:
                fail("unknown name %s" % nm)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                fail("non-numeric constant")
            v = float(node.value)
            if v != v or v in (float("inf"), float("-inf")):
                fail("non-finite constant")
    nodes = sum(1 for nd in ast.walk(tree)
                if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant)))
    if nodes > MAX_NODES:
        fail("expression too large (%d nodes)" % nodes)
    try:
        code = compile(tree, "<expr>", "eval")
    except Exception:
        fail("compile error")
    return code, nodes


def roll(code, y):
    n = len(y)
    glob = {"__builtins__": {}}
    preds = {}
    for q in range(R, n - R):
        env = dict(ALLOWED_FUNCS)
        for nm, off in TAP_NAMES.items():
            env[nm] = y[q + off]
        try:
            p = eval(code, glob, env)
        except Exception:
            fail("evaluation error")
        if not isinstance(p, (int, float)) or isinstance(p, bool):
            fail("non-numeric result")
        p = float(p)
        if p != p or p in (float("inf"), float("-inf")):
            fail("non-finite result")
        preds[q] = p
    return preds


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            header = fh.readline().split()
        t = int(header[2])
    except Exception:
        fail("bad instance header")
    if t < 1 or t > 1000000:
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

    plan = plan_for(t)
    w, sigma = plan["w"], plan["sigma"]
    r_spk, r_noi = held_rngs(t)
    target_k = max(6, NHELD // 20)
    x_true = make_spikes(r_spk, NHELD, w, target_k)
    y = convolve(x_true, w, sigma, r_noi)

    preds = roll(code, y)
    if not preds:
        fail("no scoreable positions")

    se = 0.0
    zse = 0.0
    m = 0
    for q, p in preds.items():
        xv = x_true[q]
        se += (p - xv) ** 2
        zse += xv ** 2
        m += 1
    F_mse = se / m
    B_mse = zse / m

    F = F_mse * (1.0 + LAMBDA * nodes)
    B = B_mse * (1.0 + LAMBDA * 1)
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("heldout_MSE=%.6f baseline_MSE=%.6f nodes=%d w=%d sigma=%.4f  Ratio: %.6f"
          % (F_mse, B_mse, nodes, w, sigma, sc / 1000.0))


if __name__ == "__main__":
    main()
