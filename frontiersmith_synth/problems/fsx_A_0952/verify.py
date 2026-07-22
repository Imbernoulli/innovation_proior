#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the batch-prep-ledger recurrence-recovery task.

- Reads the test id from <in>'s header, then regenerates the HIDDEN batch-prep
  law (divisors A,B; modulus M; correction table g[]; per-portion constant C;
  base counts T(0),T(1)) exactly as gen.py does. The law lives ONLY here (and
  duplicated, not imported, in gen.py).
- Regenerates a HELD-OUT, deterministic sample of large "mega-order" sizes
  (genuine extrapolation, far beyond the training ledger) and computes their
  TRUE prep-op counts via the hidden recurrence.
- Parses the participant's program: a tiny two-statement DSL,
      BASE k v0 v1 ... vk       (base counts for n = 0..k)
      REC  <expr>                (applies for n > k; may reference n and
                                   recursively call T(...))
  Expressions are arithmetic (+ - *, unary -, parentheses, integer constants)
  over: `n`, `MOD(n,m)`, `TAB(MOD(n,m), v0,...,v_{m-1})`, and recursive calls
  `T(FLOORDIV(n,k))` / `T(CEILDIV(n,k))`. No Python eval/exec is used; a
  custom recursive AST evaluator computes every value exactly (Python
  integers, no floats, no rounding).
- Rolls the program forward (memoized, cycle- and depth-guarded) on the
  held-out sizes and scores:
      EXACT_FRAC = fraction of held-out sizes matched EXACTLY (integer ==)
      CLOSE      = mean of 1/(1 + K * relative_error) over held-out sizes
      PARSIMONY  = max(0, 1 - footprint / PARS_BUDGET), footprint = BASE size
                   + REC AST node count (a genuinely open-ended axis: no
                   submission is forced to spend the full node/BASE budget)
      Ratio = clip(FLOOR + ALPHA*EXACT_FRAC + GAMMA*CLOSE + DELTA*PARSIMONY, 0, 1)
  Exact recovery of the recurrence dominates the score; a merely CLOSE
  smooth/asymptotic fit earns only the small CLOSE credit and stays well
  below a genuine recursive recovery. The formula's own ceiling sits above
  what any single reference submission achieves, since PARSIMONY keeps
  rewarding a leaner encoding of the SAME recovered law.
"""
import sys, ast, random, math, hashlib

sys.setrecursionlimit(20000)

PAIRS = [(2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (2, 6), (3, 6), (4, 5)]

FLOOR = 0.04
ALPHA = 0.55
GAMMA = 0.12
DELTA = 0.08              # weight on the PARSIMONY bonus
PARS_BUDGET = 60.0        # footprint budget for the parsimony bonus
K_CLOSE = 7.0

MAX_K = 2000            # max BASE table index (hard feasibility gate)
MAX_M = 30               # max MOD/TAB modulus
MAX_DIV = 12              # max FLOORDIV/CEILDIV divisor
MAX_NODES = 80            # max AST node count in REC expr
MAX_OUT_BYTES = 300000
MAX_CONST_ABS = 10 ** 9   # bound on any integer literal
MAX_DEPTH = 300            # recursion-chain depth guard per held-out point
MAX_MEMO = 400000          # cap on distinct n values memoized

NUM_FAR = 110


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden batch-prep law (identical to gen.py) ----------
def _law_seed(t):
    """Non-linear per-test-id seed (SHA-256 of a salted tag) -- identical to gen.py's,
    never printed, never derivable from a simple arithmetic guess."""
    tag = "fsx_A_0952-batchprep-law-v1:%d" % t
    return int.from_bytes(hashlib.sha256(tag.encode()).digest()[:8], "big")


def truth_params(t):
    rng = random.Random(_law_seed(t))
    a, b = rng.choice(PAIRS)
    m = rng.choice([4, 5, 6, 7, 8])
    g = [rng.randint(-5, 5) for _ in range(m)]
    c = rng.randint(1, 3)
    base0 = rng.randint(1, 9)
    base1 = rng.randint(1, 9)
    return a, b, m, g, c, base0, base1


def _sample_seed(t):
    tag = "fsx_A_0952-farsample-v1:%d" % t
    return int.from_bytes(hashlib.sha256(tag.encode()).digest()[:8], "big")


def far_sample(t, lo, hi, num):
    rng = random.Random(_sample_seed(t))
    lg_lo, lg_hi = math.log(lo), math.log(hi)
    seen = set()
    tries = 0
    while len(seen) < num and tries < num * 20:
        tries += 1
        u = rng.uniform(lg_lo, lg_hi)
        v = int(round(math.exp(u)))
        v = max(lo, min(hi, v))
        seen.add(v)
    return sorted(seen)


def true_T_far(ns, a, b, m, g, c, base0, base1):
    memo = {0: base0, 1: base1}

    def T(n):
        if n in memo:
            return memo[n]
        lo = n // a
        hi = -(-n // b)
        val = T(lo) + T(hi) + g[n % m] + c * n
        memo[n] = val
        return val

    sys.setrecursionlimit(10000)
    return {n: T(n) for n in ns}


# ---------- DSL parsing (custom recursive-descent AST validator) ----------
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.USub, ast.UAdd,
)


def _is_int_const(node):
    """True for a bare integer literal OR a unary-minus-wrapped integer literal."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int) and not isinstance(node.value, bool)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return _is_int_const(node.operand)
    return False


def _int_const_value(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_int_const_value(node.operand)
    raise ValueError("not an int const")


def _count_nodes(tree):
    return sum(1 for nd in ast.walk(tree)
               if isinstance(nd, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant)))


def _validate_rec_expr(text):
    """Validate the REC expression under the fixed grammar; return (tree, nodecount)."""
    text = text.strip()
    if not text:
        fail("empty REC expression")
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        fail("REC parse error")
    call_func_ids = {id(nd.func) for nd in ast.walk(tree) if isinstance(nd, ast.Call)}
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            fail("disallowed syntax %s in REC" % type(node).__name__)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, int) or isinstance(node.value, bool):
                fail("non-integer constant in REC")
            if abs(node.value) > MAX_CONST_ABS:
                fail("constant magnitude too large in REC")
        if isinstance(node, ast.Name) and id(node) not in call_func_ids:
            if node.id != "n":
                fail("unknown name '%s' in REC (only 'n' is a bare variable)" % node.id)
    _validate_calls(tree)
    nodes = _count_nodes(tree)
    if nodes > MAX_NODES:
        fail("REC expression too large (%d nodes)" % nodes)
    return tree, nodes


def _validate_calls(tree):
    """Structurally validate every Call node under the fixed function grammar:
    FLOORDIV(n,k) / CEILDIV(n,k) ; MOD(n,m) ; TAB(MOD(n,m), v0..v_{m-1}) ; T(x)."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name):
            fail("disallowed call target in REC")
        fn = node.func.id
        if node.keywords:
            fail("keyword args not allowed in REC")
        args = node.args
        if fn in ("FLOORDIV", "CEILDIV"):
            if len(args) != 2 or not (isinstance(args[0], ast.Name) and args[0].id == "n") \
               or not _is_int_const(args[1]):
                fail("%s must be %s(n, <int constant>)" % (fn, fn))
            k = _int_const_value(args[1])
            if not (2 <= k <= MAX_DIV):
                fail("%s divisor out of range [2,%d]" % (fn, MAX_DIV))
        elif fn == "MOD":
            if len(args) != 2 or not (isinstance(args[0], ast.Name) and args[0].id == "n") \
               or not _is_int_const(args[1]):
                fail("MOD must be MOD(n, <int constant>)")
            mm = _int_const_value(args[1])
            if not (2 <= mm <= MAX_M):
                fail("MOD modulus out of range [2,%d]" % MAX_M)
        elif fn == "TAB":
            if len(args) < 2:
                fail("TAB needs a MOD(...) index plus value list")
            idx = args[0]
            if not (isinstance(idx, ast.Call) and isinstance(idx.func, ast.Name)
                    and idx.func.id == "MOD"):
                fail("TAB's first argument must be MOD(n, m)")
            if len(idx.args) != 2 or not (isinstance(idx.args[0], ast.Name) and idx.args[0].id == "n") \
               or not _is_int_const(idx.args[1]):
                fail("TAB's MOD(...) must be MOD(n, <int constant>)")
            mm = _int_const_value(idx.args[1])
            if not (2 <= mm <= MAX_M):
                fail("TAB modulus out of range [2,%d]" % MAX_M)
            vals = args[1:]
            if len(vals) != mm:
                fail("TAB needs exactly %d value args to match its modulus" % mm)
            for v in vals:
                if not _is_int_const(v):
                    fail("TAB values must be integer constants")
        elif fn == "T":
            if len(args) != 1:
                fail("T must take exactly one argument")
            sub = args[0]
            if not (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                    and sub.func.id in ("FLOORDIV", "CEILDIV")):
                fail("T(...) argument must be FLOORDIV(n,k) or CEILDIV(n,k)")
        else:
            fail("unknown function '%s' in REC" % fn)


class PointFail(Exception):
    pass


def _eval(node, n, T_predict, stack, depth):
    if isinstance(node, ast.Expression):
        return _eval(node.body, n, T_predict, stack, depth)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return n
    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand, n, T_predict, stack, depth)
        if isinstance(node.op, ast.USub):
            return -v
        return v
    if isinstance(node, ast.BinOp):
        l = _eval(node.left, n, T_predict, stack, depth)
        r = _eval(node.right, n, T_predict, stack, depth)
        if isinstance(node.op, ast.Add):
            return l + r
        if isinstance(node.op, ast.Sub):
            return l - r
        if isinstance(node.op, ast.Mult):
            return l * r
        raise PointFail("bad binop")
    if isinstance(node, ast.Call):
        fn = node.func.id
        if fn == "FLOORDIV":
            k = _int_const_value(node.args[1])
            return n // k
        if fn == "CEILDIV":
            k = _int_const_value(node.args[1])
            return -(-n // k)
        if fn == "MOD":
            m = _int_const_value(node.args[1])
            return n % m
        if fn == "TAB":
            r = _eval(node.args[0], n, T_predict, stack, depth)
            return _int_const_value(node.args[1 + r])
        if fn == "T":
            n2 = _eval(node.args[0], n, T_predict, stack, depth)
            return T_predict(n2, stack, depth + 1)
        raise PointFail("bad call")
    raise PointFail("bad node")


def make_predictor(base_vals, k, rec_tree):
    memo = dict(enumerate(base_vals))

    def T_predict(n, stack=None, depth=0):
        if n in memo:
            return memo[n]
        if stack is None:
            stack = frozenset()
        if n in stack:
            raise PointFail("cycle at n=%d" % n)
        if depth > MAX_DEPTH:
            raise PointFail("max depth exceeded")
        if len(memo) > MAX_MEMO:
            raise PointFail("memo cap exceeded")
        val = _eval(rec_tree, n, T_predict, stack | {n}, depth)
        memo[n] = val
        return val

    return T_predict


def parse_program(raw):
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        fail("empty program")
    base_vals = None
    rec_tree = None
    rec_nodes = 0
    seen_base = seen_rec = False
    for ln in lines:
        head = ln.split(None, 1)
        kw = head[0].upper()
        rest = head[1] if len(head) > 1 else ""
        if kw == "BASE":
            if seen_base:
                fail("multiple BASE statements")
            seen_base = True
            toks = rest.split()
            if not toks:
                fail("BASE needs a count and values")
            try:
                k = int(toks[0])
            except ValueError:
                fail("BASE count must be an integer")
            if not (0 <= k <= MAX_K):
                fail("BASE count out of range [0,%d]" % MAX_K)
            vals_tok = toks[1:]
            if len(vals_tok) != k + 1:
                fail("BASE needs exactly k+1 = %d values" % (k + 1))
            vals = []
            for vt in vals_tok:
                try:
                    v = int(vt)
                except ValueError:
                    fail("BASE value '%s' is not an integer" % vt)
                if abs(v) > MAX_CONST_ABS:
                    fail("BASE value magnitude too large")
                vals.append(v)
            base_vals = vals
        elif kw == "REC":
            if seen_rec:
                fail("multiple REC statements")
            seen_rec = True
            rec_tree, rec_nodes = _validate_rec_expr(rest)
        else:
            fail("unknown statement '%s'" % kw)
    if not seen_base:
        fail("missing BASE statement")
    if not seen_rec:
        fail("missing REC statement")
    return base_vals, rec_tree, rec_nodes


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            header = fh.readline().split()
        n_train = int(header[0])
        t = int(header[1])
    except Exception:
        fail("bad instance header")
    if t < 1 or t > 100000 or n_train < 1:
        fail("bad test id / n_train")

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace")

    base_vals, rec_tree, rec_nodes = parse_program(text)
    k = len(base_vals) - 1
    footprint = len(base_vals) + rec_nodes
    parsimony = max(0.0, 1.0 - footprint / PARS_BUDGET)

    a, b, m, g, c, base0, base1 = truth_params(t)

    far_lo = 20000
    far_hi = min(2000000, 200000 * t)
    if far_hi <= far_lo:
        far_hi = far_lo + 20000
    ns = far_sample(t, far_lo, far_hi, NUM_FAR)
    if len(ns) < 20:
        fail("internal: insufficient held-out sample")
    truth = true_T_far(ns, a, b, m, g, c, base0, base1)

    T_predict = make_predictor(base_vals, k, rec_tree)

    n_exact = 0
    close_sum = 0.0
    total = len(ns)
    for n in ns:
        tv = truth[n]
        try:
            pv = T_predict(n)
        except PointFail:
            continue  # counts as a miss (contributes 0 to exact and close)
        except RecursionError:
            continue
        if not isinstance(pv, int) or isinstance(pv, bool):
            fail("non-integer prediction produced")
        if pv == tv:
            n_exact += 1
            close_sum += 1.0
        else:
            # pv/tv are exact Python ints of unbounded size; a grammar-legal but
            # absurdly large prediction (e.g. many chained huge constants) can
            # overflow a float conversion -- treat that as "maximally wrong"
            # (0 close credit) rather than letting the checker crash.
            try:
                rel = abs(pv - tv) / max(1, abs(tv))
                close_sum += 1.0 / (1.0 + K_CLOSE * rel)
            except (OverflowError, ValueError, ZeroDivisionError):
                pass

    exact_frac = n_exact / total
    close = close_sum / total

    raw_score = FLOOR + ALPHA * exact_frac + GAMMA * close + DELTA * parsimony
    ratio = max(0.0, min(1.0, raw_score))

    print("exact_frac=%.4f close=%.4f parsimony=%.4f footprint=%d far_range=[%d,%d] n_far=%d  Ratio: %.6f"
          % (exact_frac, close, parsimony, footprint, far_lo, far_hi, total, ratio))


if __name__ == "__main__":
    main()
