#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for "hidden proportion law of nested frames"
(family: proportion-cascade-recurrence; mechanisms: nested-scale-extrapolation
+ recurrence-recovery).

- Reads the test id from <in>'s header, and the mean noisy T_obs of the
  training rows (used ONLY to build the internal constant baseline).
- Regenerates the SAME hidden dynamical system (alpha, beta, lambda1,
  lambda2, p*) that the training rows in <in> were drawn from, purely from
  the test id -- identical code path to gen.py, so the law matches exactly.
  The ground truth lives ONLY here; it is never printed anywhere.
- Regenerates a HELD-OUT batch of DEEP designs (depth 10..14), never shown
  to the solver, with the TRUE (noiseless) tension score T.
- Parses the participant's output: ONE line holding a pure arithmetic Python
  expression over p1, p2, d (+ - * / ** parentheses and numeric literals
  only; no function calls, no other names). Evaluates it on every held-out
  design.
- Scores (minimisation) by held-out MSE against a constant-mean baseline.
"""
import sys, ast, random

LAM_BUCKETS = {
    1: (1.05, 1.10), 2: (1.05, 1.10),
    3: (1.10, 1.16), 4: (1.10, 1.16),
    5: (1.17, 1.25), 6: (1.17, 1.25),
    7: (1.26, 1.34), 8: (1.26, 1.34),
    9: (1.35, 1.44), 10: (1.35, 1.44),
}
MAX_OUT_BYTES = 20000
N_HOLDOUT = 420
ALLOWED_NAMES = {"p1", "p2", "d"}
ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
    ast.Name, ast.Load,
)


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


def derive_law(t):
    lo, hi = LAM_BUCKETS.get(t, (0.55, 1.48))
    rng = random.Random("pchidden_law_v1_%d" % t)
    for _ in range(4000):
        lam1 = rng.uniform(lo, hi)
        cap = min(0.50, lam1 - 0.08)
        if cap <= 0.05:
            continue
        lam2_mag = rng.uniform(0.05, cap)
        lam2 = rng.choice([1, -1]) * lam2_mag
        if abs(lam1 - lam2) < 0.08:
            continue
        pstar = rng.uniform(0.45, 0.75)
        alpha = lam1 + lam2
        beta = -lam1 * lam2
        return {"lam1": lam1, "lam2": lam2, "alpha": alpha, "beta": beta, "pstar": pstar}
    raise RuntimeError("law derivation failed")


def true_T(p1, p2, d, alpha, beta, pstar):
    devs = [p1 - pstar, p2 - pstar]
    for _ in range(3, d + 1):
        devs.append(alpha * devs[-1] + beta * devs[-2])
    return devs[-1] * devs[-1]


GRADE_NOISE_K = 0.62  # irreducible measurement noise on the GRADED tension
                       # reading, sized off sqrt(B) -- the checker's OWN
                       # constant-baseline error -- rather than off the
                       # held-out population spread. In strongly-converging
                       # regimes the deep designs can end up nearly
                       # indistinguishable from each other (tiny population
                       # variance) even though the constant TRAIN-window
                       # baseline is still far off (large B); scaling off B
                       # guarantees F cannot collapse to ~0 relative to B no
                       # matter how tight the held-out spread is, so even a
                       # perfect recurrence-recovery leaves real headroom.


def held_out(t, law, m, mean_tobs):
    rng = random.Random("pchidden_holdout_v1_%d" % t)
    pstar = law["pstar"]
    pts = []
    for _ in range(m):
        d = rng.randint(10, 14)
        p1 = min(0.98, max(0.05, pstar + rng.uniform(-0.30, 0.30)))
        p2 = min(0.98, max(0.05, pstar + rng.uniform(-0.30, 0.30)))
        T = true_T(p1, p2, d, law["alpha"], law["beta"], pstar)
        pts.append((p1, p2, d, T))
    B_true = sum((mean_tobs - T) ** 2 for _, _, _, T in pts) / len(pts)
    noise_sd = GRADE_NOISE_K * (B_true ** 0.5)
    rows = []
    for p1, p2, d, T in pts:
        T_graded = max(0.0, T + rng.gauss(0.0, noise_sd))
        rows.append((p1, p2, d, T_graded))
    return rows


def parse_instance(inf):
    with open(inf) as fh:
        head = fh.readline().split()
        if len(head) < 2:
            raise ValueError("bad header")
        n = int(head[0]); t = int(head[1])
        tobs = []
        for _ in range(n):
            parts = fh.readline().split()
            if not parts:
                raise ValueError("missing row")
            tobs.append(float(parts[-1]))
    if n <= 0:
        raise ValueError("no rows")
    return t, tobs


def safe_compile(text):
    text = text.strip()
    if not text:
        fail("empty expression")
    if len(text) > 4000:
        fail("expression too long")
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error")
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            fail("disallowed syntax %s" % type(node).__name__)
        if isinstance(node, ast.Name) and node.id not in ALLOWED_NAMES:
            fail("unknown name %s" % node.id)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                fail("non-numeric constant")
            v = float(node.value)
            if v != v or v in (float("inf"), float("-inf")):
                fail("non-finite constant")
    try:
        return compile(tree, "<expr>", "eval")
    except Exception:
        fail("compile error")


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        t, tobs = parse_instance(inf)
    except Exception:
        fail("bad instance file")
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
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        fail("empty output")
    code = safe_compile(lines[0])

    mean_tobs = sum(tobs) / len(tobs)
    law = derive_law(t)
    rows = held_out(t, law, N_HOLDOUT, mean_tobs)

    se = 0.0
    for p1, p2, d, T in rows:
        try:
            val = eval(code, {"__builtins__": {}}, {"p1": p1, "p2": p2, "d": d})
        except Exception:
            fail("evaluation error")
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            fail("non-numeric result")
        val = float(val)
        if val != val or val in (float("inf"), float("-inf")):
            fail("non-finite result")
        se += (val - T) ** 2
    F = se / len(rows)

    B = sum((mean_tobs - T) ** 2 for _, _, _, T in rows) / len(rows)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("heldout_MSE=%.6f baseline_MSE=%.6f  Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
