#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the ghost-train timetable prefetcher.

- Reads (N, P, testId) from <in>'s header, then regenerates the FULL hidden
  timetable (identical formula to gen.py) purely from testId.  Training
  covered epochs r = 0..Rtrain-1; the grading queries below live at epochs
  r = Rtrain+1 .. Rtrain+H (strictly beyond anything the solver saw) -- the
  hidden law lives ONLY here, never in <in>.
- Parses the participant's TWO-SLOT prefetch predictor:
      SLOT1 <expr>
      SLOT2 <expr> | NONE
  <expr> is a restricted arithmetic expression over t, h1, h2, h3, h4
  (h1 = platform at tick t-1 ... h4 = platform at tick t-4): + - * // % **,
  parentheses, integer constants with |c| <= 1e6, unary +/-.  ** exponents
  must be a constant literal in [0,4].  Anything else -> Ratio: 0.0.
- For each held-out query the true recent history h1..h4 is handed to the
  predictor (teacher forced); it must name up to 2 candidate platforms.
  credit = 1/(1+|slot1-true|/SCALE), or 0.7/(1+|slot2-true|/SCALE) if that
  beats slot1 (slot2 costs a budget discount).  F = mean credit over the H
  queries.  A single-slot order-1 local predictor (2*h1-h2, computed by the
  grader itself, never told the epoch law) is the internal baseline B.
      Ratio = min(1000, 100*F/B) / 1000
  A predictor that recovers B0,B1,B2,S0 -- the epoch-level drift law -- and
  extrapolates it to the unseen epochs scores far above B; one that only
  looks at recent local deltas cannot see the drift and stays near B.
"""
import sys
import ast

W = 4
H = 240
SCALE = 2.0
MAX_OUT_BYTES = 20000
MAX_NODES = 30
MAX_CONST = 10 ** 6


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden timetable (identical to gen.py) ----------
def hidden_params(tid):
    import random
    rng = random.Random(5170001 + tid * 104729)
    P = rng.randint(50, 110)
    S0 = rng.randint(3, 9)
    B0 = rng.randint(5, 60)
    B1 = rng.randint(2, 6)
    B2 = 0 if tid <= 3 else rng.randint(1, 3)
    J3 = rng.choice([3, 4, 5, 6])
    Jc = rng.randint(0, 999983)
    Rtrain = 20 + 6 * tid
    return P, S0, B0, B1, B2, J3, Jc, Rtrain


def jitter(t, J3, Jc):
    h = (t * 2654435761 + Jc) & 0xFFFFFFFF
    h ^= (h >> 13)
    h = (h * 2246822519) & 0xFFFFFFFF
    h ^= (h >> 15)
    return h % J3


def platform(t, P, S0, B0, B1, B2, J3, Jc):
    r = t // P
    return B0 + B1 * r + B2 * r * r + S0 * (t % P) + jitter(t, J3, Jc)


# ---------- restricted expression grammar ----------
_ALLOWED_NAMES = {"t", "h1", "h2", "h3", "h4"}
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
)


def _count_nodes(tree):
    return sum(1 for nd in ast.walk(tree)
               if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Name, ast.Constant)))


def compile_expr(text):
    text = text.strip()
    if not text:
        fail("empty expression")
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("parse error in expression")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            fail("disallowed syntax %s" % type(node).__name__)
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_NAMES:
            fail("unknown identifier '%s'" % node.id)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, int) or isinstance(node.value, bool):
                fail("non-integer constant")
            if abs(node.value) > MAX_CONST:
                fail("constant out of range")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            rhs = node.right
            if not (isinstance(rhs, ast.Constant) and isinstance(rhs.value, int)
                    and not isinstance(rhs.value, bool) and 0 <= rhs.value <= 4):
                fail("** exponent must be a constant in [0,4]")
    if _count_nodes(tree) > MAX_NODES:
        fail("expression too large")
    try:
        code = compile(tree, "<expr>", "eval")
    except Exception:
        fail("compile error in expression")
    return code


def parse_program(text):
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        fail("need SLOT1 and SLOT2 lines")
    l1, l2 = lines[0].strip(), lines[1].strip()
    if not l1.startswith("SLOT1"):
        fail("first line must start with SLOT1")
    if not l2.startswith("SLOT2"):
        fail("second line must start with SLOT2")
    e1 = l1[len("SLOT1"):].strip()
    e2 = l2[len("SLOT2"):].strip()
    code1 = compile_expr(e1)
    code2 = None if e2 == "NONE" else compile_expr(e2)
    return code1, code2


def safe_eval(code, env):
    if code is None:
        return None
    try:
        v = eval(code, {"__builtins__": {}}, env)
    except Exception:
        return None
    if not isinstance(v, int) or isinstance(v, bool):
        return None
    if abs(v) > 10 ** 15:
        return None
    return v


def credit(pred1, pred2, true):
    def cr(p):
        if p is None:
            return 0.0
        return 1.0 / (1.0 + abs(p - true) / SCALE)
    c1 = cr(pred1)
    c2 = 0.7 * cr(pred2) if pred2 is not None else 0.0
    return max(c1, c2)


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            header = fh.readline().split()
        N_in, P_in, tid = int(header[0]), int(header[1]), int(header[2])
    except Exception:
        fail("bad instance header")
    if tid < 1 or tid > 100000:
        fail("bad test id")

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace")

    code1, code2 = parse_program(text)

    P, S0, B0, B1, B2, J3, Jc, Rtrain = hidden_params(tid)
    if P != P_in or P * Rtrain != N_in:
        fail("instance/header mismatch")

    def addr(tt):
        return platform(tt, P, S0, B0, B1, B2, J3, Jc)

    total_F = 0.0
    total_B = 0.0
    for j in range(H):
        r = Rtrain + 1 + j
        off = j % W
        t = r * P + off
        h = [addr(t - k) for k in range(1, W + 1)]
        true = addr(t)
        env = {"t": t, "h1": h[0], "h2": h[1], "h3": h[2], "h4": h[3]}
        p1 = safe_eval(code1, env)
        p2 = safe_eval(code2, env)
        total_F += credit(p1, p2, true)
        # internal baseline: order-1 local delta, no epoch-law knowledge
        base_pred = h[0] + (h[0] - h[1])
        total_B += credit(base_pred, None, true)

    F = total_F / H
    B = total_B / H
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.6f baseline=%.6f  Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
