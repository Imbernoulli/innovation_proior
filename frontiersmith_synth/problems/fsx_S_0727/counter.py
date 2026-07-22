import sys

# Format D checker -- "monotone register ratchet" minimal addition program.
#
#   1) Parse N, K, B0 and the N distinct positive-integer targets from <in>.
#   2) Parse the participant's straight-line program from <out>:
#         T
#         T lines (whitespace-separated, may wrap freely), each "i j" meaning
#         "create register k := reg[i] + reg[j]" where k is this op's
#         1-indexed position (registers 1..T are created in order; register 0
#         is given, value 1).
#   3) Simulate it, enforcing:
#        - append-only / no forward reference: 0 <= i,j < k (op k may only
#          reference registers 0..k-1) -- the monotone ratchet.
#        - a REUSE BUDGET per register: register 0 (the constant 1) is
#          unconstrained; a register created at position t (t = the op index
#          that created it, t>=1) may be used as an operand at most
#          B0 + t // K times in total, over the rest of the program. This
#          cap only ever grows with t (monotone, never revisited/lowered).
#        - values stay positive and bounded (sanity DoS guard).
#   4) EXACT-equivalence gate: every target value must appear in some
#      register when the program ends (else Ratio: 0.0).
#   5) Objective (minimize) = T. Baseline B = the checker's own trivial
#      per-target construction (each target built independently from
#      register 0 via the standard binary double-and-add method, which never
#      touches any budget since register 0 is unconstrained).
#      Ratio = min(1, 0.1 * B / T).

MAX_T = 20000
MAX_VAL = 10 ** 15
BIGINF = 10 ** 9


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def popcount(v):
    return bin(v).count("1")


def naive_cost(v):
    if v <= 1:
        return 0
    return (v.bit_length() - 1) + (popcount(v) - 1)


def main():
    inp = open(sys.argv[1]).read().split()
    out_text = open(sys.argv[2]).read()

    it = iter(inp)
    try:
        N = int(next(it)); K = int(next(it)); B0 = int(next(it))
    except Exception:
        fail("bad header")
    if not (1 <= N <= 200 and 1 <= K <= 100 and 0 <= B0 <= 100):
        fail("bad params")
    targets = []
    try:
        for _ in range(N):
            targets.append(int(next(it)))
    except Exception:
        fail("bad targets")
    if len(set(targets)) != N or any(v <= 0 for v in targets):
        fail("targets must be N distinct positive integers")

    B = sum(naive_cost(v) for v in targets)
    if B <= 0:
        fail("degenerate baseline")

    # ---- parse participant output ----
    toks = out_text.split()
    if not toks:
        fail("empty output")
    ti = iter(toks)
    try:
        T = int(next(ti))
    except Exception:
        fail("bad T")
    if T < 0 or T > MAX_T:
        fail("T out of range [0, %d]" % MAX_T)

    values = [1]                 # register 0
    budget = [BIGINF]            # register 0: unconstrained (free constant)

    for k in range(1, T + 1):
        try:
            i = int(next(ti)); j = int(next(ti))
        except Exception:
            fail("bad op line %d (missing/non-integer tokens)" % k)
        if not (0 <= i < k and 0 <= j < k):
            fail("op %d references register outside [0,%d) (forward/self reference)" % (k, k))
        if i == j:
            if budget[i] < 2:
                fail("ratchet budget exhausted at op %d (register %d)" % (k, i))
            budget[i] -= 2
        else:
            if budget[i] < 1 or budget[j] < 1:
                fail("ratchet budget exhausted at op %d (register %d or %d)" % (k, i, j))
            budget[i] -= 1
            budget[j] -= 1
        val = values[i] + values[j]
        if val <= 0 or val > MAX_VAL:
            fail("register value out of range at op %d" % k)
        values.append(val)
        budget.append(B0 + k // K)

    have = set(values)
    missing = [v for v in targets if v not in have]
    if missing:
        fail("missing target(s), e.g. %d" % missing[0])

    F = T
    ratio = min(1000.0, 100.0 * B / max(1e-9, F)) / 1000.0
    print("B=%d F=%d Ratio: %.6f" % (B, F, ratio))


if __name__ == "__main__":
    main()
