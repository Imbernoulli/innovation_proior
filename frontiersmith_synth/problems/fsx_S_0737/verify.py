#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the flickering-lantern-row rule-mining task.

- Reads only the test id `t` from <in>'s header, then rebuilds the hidden
  radius-1 rule, the noisy training snapshots (for cross-check only) and a
  FRESH HELD-OUT initial row + its L-tick rollout under the TRUE rule,
  entirely from `t` (function `_build`, byte-identical to gen.py's). The
  hidden rule number and the held-out trace are NEVER read from the input
  file -- they are re-derived here, so the ground truth lives only in this
  checker.
- Parses the participant's output: a SINGLE boolean expression over the
  neighbourhood variables `cL` (left neighbour), `cM` (self), `cR` (right
  neighbour), using only `and or not ^ == != ( )` and the constants `0`/`1`.
  This expression IS the candidate radius-1 update rule: it is evaluated on
  all 8 neighbourhood patterns to build an 8-entry rule table.
- The candidate rule is rolled forward from the SAME held-out clean initial
  row for L=500 ticks (long-horizon rollout) and compared, cell by cell, to
  the TRUE rule's rollout AFTER a fixed observation-noise pass (so even the
  exact rule cannot score a perfect match -- the noise floor keeps headroom).
      F = mismatch_fraction * (1 + LAMBDA * nodes)
      B = mismatch_fraction_of_the_identity_rule * (1 + LAMBDA)   # "lanterns never change"
      Ratio = min(1000, 100*B/F) / 1000
  A rule that reproduces the true dynamics exactly lands near the noise floor
  (~18% mismatch); a rule fit to the wrong composition depth drifts toward
  50% (uncorrelated) over the 500-tick rollout.
"""
import sys, ast

LAMBDA = 0.004
MAX_NODES = 150
MAX_OUT_BYTES = 200000
RULES = [142, 227, 194, 24, 130, 190, 152, 97, 159, 43]


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden instance (IDENTICAL to gen.py's _build) ----------
def _build(t):
    import random
    rulenum = RULES[(t - 1) % len(RULES)]
    W = 80 + 12 * (t - 1)
    m = 5
    rng = random.Random(900001 + t * 7919)
    maxgap = 3 + (t - 1) // 2
    gaps = [rng.randint(2, maxgap) for _ in range(m - 1)]
    times = [0]
    for g in gaps:
        times.append(times[-1] + g)
    p_train = 0.02 + 0.003 * (t - 1)
    p_final = 0.18
    L = 500
    table = [(rulenum >> idx) & 1 for idx in range(8)]

    def step(row):
        Wn = len(row)
        return [table[row[(i - 1) % Wn] * 4 + row[i] * 2 + row[(i + 1) % Wn]] for i in range(Wn)]

    def noisy(row, p):
        return [(1 - b) if rng.random() < p else b for b in row]

    train_row0 = [rng.randint(0, 1) for _ in range(W)]
    clean = train_row0[:]
    snapshots = [noisy(clean, p_train)]
    for g in gaps:
        for _ in range(g):
            clean = step(clean)
        snapshots.append(noisy(clean, p_train))

    grade_row0 = [rng.randint(0, 1) for _ in range(W)]
    g_true = grade_row0[:]
    for _ in range(L):
        g_true = step(g_true)
    g_observed_final = noisy(g_true, p_final)

    return dict(rulenum=rulenum, W=W, times=times, snapshots=snapshots,
                p_train=p_train, p_final=p_final, L=L,
                grade_row0=grade_row0, g_observed_final=g_observed_final,
                table=table)


# ---------- expression grammar ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BoolOp, ast.And, ast.Or,
    ast.UnaryOp, ast.Not,
    ast.BinOp, ast.BitXor,
    ast.Compare, ast.Eq, ast.NotEq,
    ast.Name, ast.Load, ast.Constant,
)
_ALLOWED_NAMES = ("cL", "cM", "cR")


def _validate_ast(tree):
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return "disallowed syntax %s" % type(node).__name__
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_NAMES:
            return "unknown name %s" % node.id
        if isinstance(node, ast.Compare) and (len(node.ops) != 1 or len(node.comparators) != 1):
            return "chained comparison not allowed"
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                pass
            elif isinstance(node.value, int):
                if node.value not in (0, 1):
                    return "constant out of range"
            else:
                return "non-integer constant"
    return None


def _count_nodes(tree):
    return sum(1 for nd in ast.walk(tree)
               if isinstance(nd, (ast.BoolOp, ast.UnaryOp, ast.BinOp, ast.Compare,
                                   ast.Name, ast.Constant)))


def parse_rule(raw):
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        fail("empty program")
    text = lines[0]
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
        code = compile(tree, "<rule>", "eval")
    except Exception:
        fail("compile error")

    table = [None] * 8
    glob = {"__builtins__": {}}
    for cl in (0, 1):
        for cm in (0, 1):
            for cr in (0, 1):
                env = {"cL": cl, "cM": cm, "cR": cr}
                try:
                    v = eval(code, glob, env)
                except Exception:
                    fail("evaluation error")
                if isinstance(v, bool):
                    v = int(v)
                if not isinstance(v, int) or v not in (0, 1):
                    fail("rule output not in {0,1}")
                table[cl * 4 + cm * 2 + cr] = v
    return table, nodes


# ---------- rollout ----------
def roll(table, row0, L):
    row = list(row0)
    W = len(row)
    for _ in range(L):
        row = [table[row[(i - 1) % W] * 4 + row[i] * 2 + row[(i + 1) % W]] for i in range(W)]
    return row


def mismatch_fraction(a, b):
    W = len(a)
    return sum(1 for x, y in zip(a, b) if x != y) / float(W)


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]
    try:
        header = open(inf).readline().split()
        t = int(header[0])
    except Exception:
        fail("bad input file")

    try:
        raw = open(outf, "r", errors="replace").read()
    except Exception:
        fail("cannot read output")

    table, nodes = parse_rule(raw)

    d = _build(t)
    grade_row0, g_observed_final, L = d["grade_row0"], d["g_observed_final"], d["L"]

    predicted_final = roll(table, grade_row0, L)
    mf = mismatch_fraction(predicted_final, g_observed_final)
    F = mf * (1.0 + LAMBDA * nodes)

    baseline_final = grade_row0  # identity rule: the row never changes
    mf_base = mismatch_fraction(baseline_final, g_observed_final)
    B = mf_base * (1.0 + LAMBDA * 1.0)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("mismatch=%.6f nodes=%d Ratio: %.6f" % (mf, nodes, sc / 1000.0))


if __name__ == "__main__":
    main()
