#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans is an unused placeholder, per format-C contract)

Deterministic scorer for "Three Calls Deep" (api-fuzz-schedule).

Instance (stdin, from gen.py):
  T C budget
  repeated T times (type label 0..T-1, in label order):
    k dep_1 .. dep_k          (dep_i are other type labels; k may be 0)
    S_t                       (number of local states 0..S_t-1; state 0 is initial)
    S_t lines of C ints       (trans[t][s][c] = next state for opcode c)

Artifact (stdout): a call sequence.
  M
  M lines, each either
    C t        (attempt to create a fresh resource of type t)
    O r c      (apply opcode c to the resource created by call-line r,
                0-indexed among the M lines, 0 <= r < M)

All 1 + (sum of per-line token counts) tokens are read positionally.

Feasibility (strict -> Ratio 0.0 on any violation):
  * M parses as int, 0 <= M <= budget.
  * exactly M call-lines follow; no extra/missing tokens.
  * each call-line's first token is exactly "C" or "O" (nothing else).
  * "C t": t parses as int, 0 <= t < T.
  * "O r c": r parses as int, 0 <= r < M; c parses as int, 0 <= c < C.
    (r need not point at a line that actually produced a live resource by
    this point -- that is a SEMANTIC no-op, not a format violation: exactly
    like a real fuzzer harness getting a 404 for an object handle that
    doesn't exist yet. r merely has to be a syntactically declarable line
    index, so garbage/huge/negative/non-finite tokens are still rejected.)

Simulation:
  * "C t": succeeds (a NEW resource id is bound to this line, state 0) iff
    every type in deps[t] already has >=1 successfully-created resource
    (from ANY earlier line, tracked in a monotone "has-instance" set).
    Otherwise: no-op, this line never produces a live resource.
  * "O r c": if line r produced a live resource R, apply
    R.state = trans[R.type][R.state][c] (this may be a self-loop / no
    progress). Otherwise: no-op.
  * Every time a resource of type t reaches a state s (s can be 0, but s=0
    is free / already "owned" and never scored), record (t, s) as covered.

Objective (a weighted, first-time-only COVERAGE metric -- deeper local
states are worth quadratically more, so it rewards *walking* a resource's
own state chain, not merely touching many resources shallowly):
  F = sum over every (type, state) pair EVER covered (state >= 1) of state^2

Baseline B: the checker's own naive, dependency-oblivious construction --
attempt "C" for every label 0..T-1 once, in raw label order (ignoring
deps[] entirely), then spend any leftover budget cycling opcodes
0,1,...,C-1,0,... round-robin across call-line indices 0..T-1 (whichever of
those happened to succeed). This always nets at least the always-creatable
types (deps==[] unconditionally, e.g. every scaffold root and every shallow
type) but discovers depth-3 states only by luck of the permutation +
opcode cycle -- exactly the "recipe, no dependency graph" baseline.

  Ratio = min(1.0, 100 * F / (10 * max(1, B))) / 1.0   i.e.
  sc = min(1000.0, 100.0 * F / max(1e-9, B));  print Ratio = sc / 1000.0

O(M + total transition-table size); pure integer arithmetic; deterministic.
"""
import sys


def fail(msg):
    print(f"INVALID: {msg}")
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    with open(path, "r") as f:
        toks = f.read().split()
    it = iter(toks)

    def nxt():
        return next(it)

    T = int(nxt()); C = int(nxt()); budget = int(nxt())
    if T <= 0 or C <= 0 or budget < 0:
        raise ValueError("bad header")
    deps = [None] * T
    tables = [None] * T
    for t in range(T):
        k = int(nxt())
        if k < 0:
            raise ValueError("bad dep count")
        d = [int(nxt()) for _ in range(k)]
        for x in d:
            if not (0 <= x < T):
                raise ValueError("dep out of range")
        S = int(nxt())
        if S <= 0:
            raise ValueError("bad state count")
        table = []
        for s in range(S):
            row = [int(nxt()) for _ in range(C)]
            for v in row:
                if not (0 <= v < S):
                    raise ValueError("transition target out of range")
            table.append(row)
        deps[t] = d
        tables[t] = table
    return T, C, budget, deps, tables


def simulate(T, C, budget, deps, tables, calls, M):
    """calls: list of M tuples ('C', t) or ('O', r, c). Returns F (int)."""
    has_instance = [False] * T
    # line_resource[i] = (type, state_ref) or None ; state_ref is a 1-elem
    # list so we can mutate in place.
    line_resource = [None] * M
    covered = set()

    for i, call in enumerate(calls):
        if call[0] == 'C':
            t = call[1]
            if all(has_instance[dp] for dp in deps[t]):
                line_resource[i] = [t, 0]
                has_instance[t] = True
                # state 0 is free, never scored
        else:
            _, r, c = call
            res = line_resource[r] if 0 <= r < i else None
            if res is None:
                continue
            t, s = res
            ns = tables[t][s][c]
            res[1] = ns
            if ns >= 1:
                covered.add((t, ns))

    F = sum(s * s for (_, s) in covered)
    return F


def baseline(T, C, budget, deps, tables):
    """The checker's own trivial reference: touch ONLY the types that need
    no dependency reasoning at all (deps[t] == []), one op each, taking
    whichever single opcode yields the most immediate value. This is a
    genuinely feasible construction (cost 2 calls per such type, always
    << budget) that deliberately does NOT chase any multi-step chain and
    does NOT retry/plan around deps[] -- it is a closed-form, budget- and
    permutation-independent floor, so it stays a stable anchor across the
    whole test ladder instead of an accident of round-robin phase luck."""
    total = 0
    for t in range(T):
        if deps[t]:
            continue
        best = 0
        for c in range(C):
            ns = tables[t][0][c]
            if ns >= 1:
                best = max(best, ns * ns)
        total += best
    return max(total, 0), 0


def main():
    if len(sys.argv) < 3:
        fail("usage: verify.py <in> <out> <ans>")
    in_path, out_path = sys.argv[1], sys.argv[2]

    try:
        T, C, budget, deps, tables = read_instance(in_path)
    except Exception as e:
        print(f"BAD_INPUT: {e}")
        print("Ratio: 0.0")
        sys.exit(0)

    try:
        with open(out_path, "r") as f:
            raw = f.read().split()
    except Exception:
        fail("cannot read output")

    if not raw:
        fail("empty output")

    it = iter(raw)

    def nxt_tok():
        try:
            return next(it)
        except StopIteration:
            return None

    m_tok = nxt_tok()
    try:
        M = int(m_tok)
    except (ValueError, TypeError):
        fail(f"non-integer M token: {m_tok!r}")
    except OverflowError:
        fail(f"M token out of range: {m_tok!r}")

    if M < 0 or M > budget:
        fail(f"M={M} out of range [0, {budget}]")

    calls = []
    for i in range(M):
        cmd = nxt_tok()
        if cmd is None:
            fail(f"truncated output at call {i}")
        if cmd == 'C':
            tt = nxt_tok()
            if tt is None:
                fail(f"truncated C call at line {i}")
            try:
                t = int(tt)
            except (ValueError, OverflowError):
                fail(f"non-integer/overflow type token at line {i}: {tt!r}")
            if not (0 <= t < T):
                fail(f"type t={t} out of range [0,{T}) at line {i}")
            calls.append(('C', t))
        elif cmd == 'O':
            rt = nxt_tok(); ct = nxt_tok()
            if rt is None or ct is None:
                fail(f"truncated O call at line {i}")
            try:
                r = int(rt)
            except (ValueError, OverflowError):
                fail(f"non-integer/overflow r token at line {i}: {rt!r}")
            try:
                c = int(ct)
            except (ValueError, OverflowError):
                fail(f"non-integer/overflow c token at line {i}: {ct!r}")
            if not (0 <= r < M):
                fail(f"r={r} out of range [0,{M}) at line {i}")
            if not (0 <= c < C):
                fail(f"c={c} out of range [0,{C}) at line {i}")
            calls.append(('O', r, c))
        else:
            fail(f"unknown call opcode {cmd!r} at line {i} (expected 'C' or 'O')")

    if nxt_tok() is not None:
        fail("trailing garbage tokens after declared M call-lines")

    F = simulate(T, C, budget, deps, tables, calls, M)
    B, _ = baseline(T, C, budget, deps, tables)

    if B <= 0:
        fail("degenerate baseline (should not happen)")

    sc = min(1000.0, 100.0 * F / max(1e-9, float(B)))
    sc = max(0.0, sc)
    print(f"F={F} B={B} M={M}")
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
