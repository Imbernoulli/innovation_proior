#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  ->  prints 'Ratio: <x in [0,1]>' (last line authoritative).

Deterministic exact scorer for the "line-crossing-sequence" compression problem.

Ground truth: for coprime P, Q the crossing sequence S (over alphabet {H,V}) of the
segment (0,0)-(P,Q) is computed directly by merging the two arithmetic progressions
{i/P : i=1..P-1} (label V) and {j/Q : j=1..Q-1} (label H) in increasing order (exact
integer cross-multiplication, no floating point). len(S) = P+Q-2.

The participant output is a small repetition grammar (a straight-line program over the
two-letter alphabet): R rules "id sym0 c0 sym1 c1", rule i expands to
expand(sym0)*c0 + expand(sym1)*c1, where sym is 'H'/'V' (literal) or a strictly-earlier
rule id (reference). Feasibility = the grammar is well-formed (DAG, strictly earlier
refs, in-range ids/counts) AND the ANSWER rule's expansion is EXACTLY S.

Objective (maximize): cost(grammar) = sum over ALL emitted rules of (2 + ndigits(c0) +
ndigits(c1))  [description-length units: 2 structural tokens + the decimal length of
each repeat count]. F = L / (L + cost).  The checker builds its OWN naive baseline
grammar (one rule per symbol, chained left-to-right) with cost B_cost, giving
B = L / (L + B_cost). Score = min(1.0, F / B) (equivalently the brief's
sc = min(1000, 100*F/B) / 1000 normalization).

Any feasibility violation prints 'Ratio: 0.0' (+ reason) and exits 0.
"""
import sys

MAX_RULES = 3000
COUNT_CAP = 10 ** 7
MAX_OUT_BYTES = 2_000_000
MAX_MATERIALIZE_CHARS = 8_000_000


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    P, Q = int(toks[0]), int(toks[1])
    return P, Q


def true_sequence(P, Q):
    i, j = 1, 1
    out = []
    while i < P or j < Q:
        if i < P and (j >= Q or i * Q < j * P):
            out.append('V')
            i += 1
        else:
            out.append('H')
            j += 1
    return ''.join(out)


def fail(reason):
    sys.stdout.write("reason: %s\nRatio: 0.0\n" % reason)
    sys.exit(0)


def is_uint_token(tok):
    return len(tok) > 0 and tok.isdigit()


def parse_rules(text, R):
    """Parse R rule lines strictly. Returns list of (t0,c0,t1,c1) with t as 'H'/'V' or
    int rule-id (< current index). Raises ValueError with a reason on any violation."""
    lines = [ln for ln in text if ln.strip() != '']
    if len(lines) < R:
        raise ValueError("not enough rule lines")
    rules = []
    for idx in range(R):
        toks = lines[idx].split()
        if len(toks) != 5:
            raise ValueError("rule line %d: expected 5 tokens, got %d" % (idx, len(toks)))
        id_tok, s0, c0_tok, s1, c1_tok = toks
        if not is_uint_token(id_tok) or int(id_tok) != idx:
            raise ValueError("rule line %d: id field must equal %d" % (idx, idx))

        def resolve_sym(s):
            if s in ('H', 'V'):
                return s
            if not is_uint_token(s):
                raise ValueError("rule line %d: bad symbol token %r" % (idx, s))
            ref = int(s)
            if not (0 <= ref < idx):
                raise ValueError("rule line %d: reference %d not strictly earlier" % (idx, ref))
            return ref

        t0 = resolve_sym(s0)
        t1 = resolve_sym(s1)
        if not is_uint_token(c0_tok) or not is_uint_token(c1_tok):
            raise ValueError("rule line %d: counts must be non-negative decimal integers" % idx)
        c0, c1 = int(c0_tok), int(c1_tok)
        if not (0 <= c0 <= COUNT_CAP) or not (0 <= c1 <= COUNT_CAP):
            raise ValueError("rule line %d: count out of range" % idx)
        rules.append((t0, c0, t1, c1))
    return rules, lines[R:]


def rule_cost(rules):
    return sum(2 + len(str(c0)) + len(str(c1)) for (_, c0, _, c1) in rules)


LEN_SENTINEL = 2 * 10 ** 9  # any true rule length in this problem is << this; used to
# clip DP values so adversarial outputs (huge counts stacked over many levels) can never
# blow up into astronomically large Python ints -- clipped rules simply cannot equal the
# (small) target length L and fail feasibility normally.


def lengths_dp(rules):
    L = [0] * len(rules)
    for idx, (t0, c0, t1, c1) in enumerate(rules):
        len0 = 1 if t0 in ('H', 'V') else L[t0]
        len1 = 1 if t1 in ('H', 'V') else L[t1]
        v = c0 * len0 + c1 * len1
        L[idx] = v if v < LEN_SENTINEL else LEN_SENTINEL
    return L


def materialize(rules, ans, budget):
    """Iterative (stack-based) memoized expansion of only the rules reachable from `ans`
    via edges with a positive repeat count. Enforces a total-character budget."""
    memo = {}
    total_chars = [0]
    stack = [(ans, False)]
    while stack:
        i, processed = stack.pop()
        if i in memo:
            continue
        t0, c0, t1, c1 = rules[i]
        deps = []
        if c0 > 0 and t0 not in ('H', 'V') and t0 not in memo:
            deps.append(t0)
        if c1 > 0 and t1 not in ('H', 'V') and t1 not in memo:
            deps.append(t1)
        if deps and not processed:
            stack.append((i, True))
            for d in deps:
                stack.append((d, False))
        else:
            s0 = t0 if t0 in ('H', 'V') else memo[t0]
            s1 = t1 if t1 in ('H', 'V') else memo[t1]
            piece = (s0 * c0) + (s1 * c1)
            total_chars[0] += len(piece)
            if total_chars[0] > budget:
                raise ValueError("materialization budget exceeded")
            memo[i] = piece
    return memo[ans]


def baseline_cost(S):
    """Checker's own naive per-symbol chain construction: rule0 = S[0]; rule_i =
    ref(rule_{i-1})*1 + S[i]*1. Always feasible, always exact."""
    L = len(S)
    rules = [('X', 1, 'H', 0)]  # placeholder t0, overwritten below (literal symbol)
    rules[0] = (S[0], 1, 'H', 0)
    for k in range(1, L):
        rules.append((k - 1, 1, S[k], 1))
    return rule_cost(rules)


def main():
    if len(sys.argv) < 3:
        print("usage: verify.py <in> <out> <ans>", file=sys.stderr)
        sys.exit(1)
    in_path, out_path = sys.argv[1], sys.argv[2]

    P, Q = read_instance(in_path)
    S = true_sequence(P, Q)
    L = len(S)
    if L <= 0:
        fail("degenerate instance")

    try:
        with open(out_path, 'rb') as f:
            raw = f.read(MAX_OUT_BYTES + 1)
    except OSError:
        fail("cannot read output file")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    try:
        text = raw.decode('ascii', errors='strict')
    except UnicodeDecodeError:
        fail("output not ascii")

    lines = text.split('\n')
    lines = [ln for ln in lines if ln.strip() != '']
    if not lines:
        fail("empty output")

    r_tok = lines[0].split()
    if len(r_tok) != 1 or not is_uint_token(r_tok[0]):
        fail("first line must be a single non-negative integer R")
    R = int(r_tok[0])
    if not (1 <= R <= MAX_RULES):
        fail("R out of range [1,%d]" % MAX_RULES)

    body = lines[1:]
    try:
        rules, rest = parse_rules(body, R)
    except ValueError as e:
        fail(str(e))

    if not rest:
        fail("missing ANSWER line")
    ans_tok = rest[0].split()
    if len(ans_tok) != 2 or ans_tok[0] != 'ANSWER' or not is_uint_token(ans_tok[1]):
        fail("expected 'ANSWER <ruleId>' line")
    k = int(ans_tok[1])
    if not (0 <= k < R):
        fail("ANSWER rule id out of range")

    Larr = lengths_dp(rules)
    if Larr[k] != L:
        fail("ANSWER rule expands to length %d, expected %d" % (Larr[k], L))

    try:
        got = materialize(rules, k, MAX_MATERIALIZE_CHARS)
    except ValueError as e:
        fail(str(e))

    if got != S:
        # find first mismatch for a helpful reason string (not scored on)
        m = 0
        while m < len(got) and m < len(S) and got[m] == S[m]:
            m += 1
        fail("reconstruction mismatch at index %d" % m)

    cost = rule_cost(rules)
    F = L / (L + cost)

    Bcost = baseline_cost(S)
    B = L / (L + Bcost)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    sys.stdout.write("L=%d cost=%d baseline_cost=%d Ratio: %.6f\n" % (L, cost, Bcost, sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
