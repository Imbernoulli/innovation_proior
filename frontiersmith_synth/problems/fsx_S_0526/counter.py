import sys

# Format D checker -- reversible-circuit gate-count golf ("peel the affine shell").
#   1) Parse header (n, a, costN, costC, costT) and the permutation table P[0..2^n-1].
#   2) Parse the participant's straight-line reversible circuit over n+a wires
#      (gates NOT/CNOT/TOF).  Strict schema + range + finiteness validation.
#   3) EXACT-equivalence gate: simulate on every input x (ancilla = 0); require the
#      first n wires to hold P(x) AND all ancilla wires restored to 0.  Any miss -> 0.
#   4) Objective (minimize) = weighted gate cost F = costN*#NOT+costC*#CNOT+costT*#TOF.
#      Baseline B = cost of the structure-blind compute-swap-uncompute synthesis
#      (full-table ANF of P and P^{-1}); ratio = min(1, 0.1*B/F).

MAX_GATES = 400000


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def anf_bits(F, n):
    # returns anf[j] = list of monomial masks (ANF coeff 1) for output bit j.
    N = 1 << n
    res = []
    for j in range(n):
        a = [(F[x] >> j) & 1 for x in range(N)]
        for i in range(n):
            bit = 1 << i
            for x in range(N):
                if x & bit:
                    a[x] ^= a[x ^ bit]
        res.append([m for m in range(N) if a[m]])
    return res


def baseline_cost(P, n, cN, cC, cT):
    # inverse table
    N = 1 << n
    Pinv = [0] * N
    for x in range(N):
        Pinv[P[x]] = x
    total_C = total_L = total_Q = 0
    for F in (P, Pinv):
        anf = anf_bits(F, n)
        for j in range(n):
            for m in anf[j]:
                pc = bin(m).count("1")
                if pc == 0:
                    total_C += 1
                elif pc == 1:
                    total_L += 1
                elif pc == 2:
                    total_Q += 1
                else:
                    # degree>2 should not occur for the planted family; treat each
                    # such term as its own (expensive) unit so B stays positive.
                    total_Q += 1
    # compute + swap(3n CNOT) + uncompute
    B = cN * total_C + cC * (total_L + 3 * n) + cT * total_Q
    return B


def main():
    try:
        inp = open(sys.argv[1]).read().split()
    except Exception:
        fail("cannot read input")
    it = iter(inp)
    try:
        n = int(next(it)); a = int(next(it))
        cN = int(next(it)); cC = int(next(it)); cT = int(next(it))
    except Exception:
        fail("bad header")
    if not (1 <= n <= 12) or a < 0:
        fail("bad n/a")
    N = 1 << n
    W = n + a
    P = [0] * N
    try:
        for x in range(N):
            P[x] = int(next(it))
    except Exception:
        fail("bad table")
    if sorted(P) != list(range(N)):
        fail("input table not a permutation")

    # ---- parse participant circuit ----
    try:
        toks = open(sys.argv[2]).read().split("\n")
    except Exception:
        fail("cannot read output")

    gates = []
    nN = nC = nT = 0
    for raw in toks:
        s = raw.strip()
        if not s:
            continue
        if len(gates) > MAX_GATES:
            fail("too many gates")
        parts = s.split()
        op = parts[0].upper()
        try:
            args = [int(v) for v in parts[1:]]
        except Exception:
            fail("non-integer / non-finite gate arg")
        for w in args:
            if w < 0 or w >= W:
                fail("wire index out of range")
        if op == "NOT" and len(args) == 1:
            gates.append((0, args[0], -1, -1)); nN += 1
        elif op == "CNOT" and len(args) == 2:
            if args[0] == args[1]:
                fail("CNOT control==target")
            gates.append((1, args[1], args[0], -1)); nC += 1
        elif op == "TOF" and len(args) == 3:
            if len({args[0], args[1], args[2]}) != 3:
                fail("TOF wires not distinct")
            gates.append((2, args[2], args[0], args[1])); nT += 1
        else:
            fail("bad gate: %s" % s)

    # ---- exact-equivalence simulation over all inputs ----
    datamask = (1 << n) - 1
    for x in range(N):
        st = x  # ancilla bits above n start at 0
        for (kind, tgt, c0, c1) in gates:
            if kind == 0:
                st ^= (1 << tgt)
            elif kind == 1:
                if (st >> c0) & 1:
                    st ^= (1 << tgt)
            else:
                if ((st >> c0) & 1) and ((st >> c1) & 1):
                    st ^= (1 << tgt)
        if (st & datamask) != P[x]:
            fail("circuit does not realize P at x=%d" % x)
        if (st >> n) != 0:
            fail("ancilla not restored at x=%d" % x)

    F = cN * nN + cC * nC + cT * nT
    if F <= 0:
        # identity permutation with empty circuit: give the baseline reference.
        F = 1
    B = baseline_cost(P, n, cN, cC, cT)
    if B <= 0:
        B = 1
    ratio = min(1.0, 0.1 * B / max(1e-9, F))
    print("F=%d B=%d nT=%d nC=%d nN=%d Ratio: %.6f" % (F, B, nT, nC, nN, ratio))


if __name__ == "__main__":
    main()
