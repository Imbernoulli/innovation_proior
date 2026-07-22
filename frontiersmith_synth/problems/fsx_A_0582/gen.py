import sys, random

# gen.py <testId>  -- prints ONE "bureaucrat's arithmetic ritual" instance to stdout.
#
# The instance is a long straight-line program (SLP) over F_p in 8 fixed inputs
# x0..x7.  It *secretly* computes a tridiagonal continuant (the determinant of an
# n x n tridiagonal matrix whose diagonal is (x0,...,x_{n-1}) and whose off-diagonal
# products are fixed constants c_1..c_{n-1}).  We EMIT it in fully expanded monomial
# form (every non-adjacent subset of variables as its own product chain, no sharing)
# and then BLOAT it with an equal amount of dead scaffolding, so:
#   * echoing the ritual verbatim               -> baseline (trivial)
#   * dead-code / constant-fold / copy-prop pass -> ~2x smaller (greedy)
#   * recognising it as a continuant and writing the 3-term recurrence -> ~n/... ops (strong)
# The recurrence structure is INVISIBLE to any syntactic optimiser; recovering it
# requires black-box interpolation + closed-form identity recognition.
#
# Difficulty grows with testId (matrix size n = 4..8).  Deterministic in testId only.

P = 2147483647          # 2^31 - 1, prime
NIN = 8                 # number of program inputs x0..x7 (indices 0..7)

SPECS = {1:4, 2:4, 3:5, 4:5, 5:6, 6:6, 7:7, 8:7, 9:8, 10:8}


def continuant_monomials(n, c):
    # D_{-1}=1, D_0=x_0, D_k = x_k*D_{k-1} - c_k*D_{k-2}; multilinear in x_0..x_{n-1}.
    Dp2 = {frozenset(): 1 % P}
    Dp1 = {frozenset([0]): 1 % P}
    for k in range(1, n):
        newD = {}
        for s, v in Dp1.items():
            ns = s | {k}
            newD[ns] = (newD.get(ns, 0) + v) % P
        for s, v in Dp2.items():
            newD[s] = (newD.get(s, 0) - c[k] * v) % P
        Dp2, Dp1 = Dp1, newD
    return Dp1


def main():
    tid = int(sys.argv[1])
    n = SPECS[tid]
    rng = random.Random(918273 + 1000 * tid)
    c = {k: rng.randrange(1, P) for k in range(1, n)}
    mons = continuant_monomials(n, c)

    prog = []                      # each: ("const",v) | ("add",i,j) | ("sub",i,j) | ("mul",i,j)

    def emit(ins):
        prog.append(ins)
        return NIN + len(prog) - 1  # value index of this instruction

    def opcount():
        return sum(1 for x in prog if x[0] in ("add", "sub", "mul"))

    # ---- core: expanded monomial form (each term an independent product chain) ----
    term_idx = []
    for s in sorted(mons.keys(), key=lambda z: (len(z), sorted(z))):
        v = mons[s] % P
        if v == 0:
            continue
        idxs = sorted(s)
        if not idxs:
            term_idx.append(emit(("const", v)))
            continue
        acc = idxs[0]
        for nxt in idxs[1:]:
            acc = emit(("mul", acc, nxt))
        if v != 1 % P:
            cidx = emit(("const", v))
            acc = emit(("mul", acc, cidx))
        term_idx.append(acc)

    running = term_idx[0]
    for ti in term_idx[1:]:
        running = emit(("add", running, ti))
    core_result = running
    core_ops = opcount()

    # ---- bloat: (core_ops - 1) dead ops, then a final identity copy as the LAST line ----
    for t in range(core_ops - 1):
        a = t % n
        b = (t + 1) % n
        emit(("mul", a, b))              # never referenced -> dead
    zidx = emit(("const", 0))
    emit(("add", core_result, zidx))     # last instruction == program result

    # ---- serialize ----
    lines = [str(P), str(len(prog))]
    for x in prog:
        if x[0] == "const":
            lines.append("const %d" % (x[1] % P))
        else:
            lines.append("%s %d %d" % (x[0], x[1], x[2]))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
