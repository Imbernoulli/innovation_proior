# TIER: invalid
# Same structure as the strong solution, but the bias term is silently
# dropped from the update (a plausible off-by-one "forgot the constant"
# bug). The SLP still parses fine and is internally consistent (a DAG,
# in-range references, well-formed out line) -- it is wrong on the merits:
# for almost every substituted beta (any beta[a] > 0 that is reachable) the
# computed values diverge from the true fixed point, so the checker's
# exact-equivalence check (repeated over several independent beta trials)
# must score this 0.
import sys


def mult(a, b, n):
    return n[b] if b != a else max(0, n[a] - 1)


def read_instance():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); K = int(next(it)); CAP = int(next(it))
    W = [[int(next(it)) for _ in range(K)] for _ in range(K)]
    c = [int(next(it)) for _ in range(N)]
    return N, K, CAP, W, c


def main():
    N, K, CAP, W, c = read_instance()
    n = [0] * K
    for ci in c:
        n[ci] += 1

    nz = [[b for b in range(K) if W[a][b] != 0 and mult(a, b, n) != 0] for a in range(K)]
    R_max = K * CAP + 2

    prog = []

    def emit_const(v):
        prog.append("const %d" % v)
        return K + len(prog) - 1

    def emit_op(op, i, j):
        prog.append("%s %d %d" % (op, i, j))
        return K + len(prog) - 1

    zero = emit_const(0)
    yn = [zero for _ in range(K)]
    capc = emit_const(CAP)

    for _ in range(R_max):
        new_yn = [0] * K
        for a in range(K):
            terms = []
            for b in nz[a]:
                coef = W[a][b] * mult(a, b, n)
                cc = emit_const(coef)
                terms.append(emit_op("mul", cc, yn[b]))
            if terms:
                s = terms[0]
                for t2 in terms[1:]:
                    s = emit_op("add", s, t2)
                # BUG: bias term (community a's beta input node) intentionally
                # not added here
            else:
                s = zero
            s = emit_op("min", s, capc)
            new_yn[a] = s
        yn = new_yn

    out_line = [yn[c[i]] for i in range(N)]

    out = sys.stdout
    out.write("%d\n" % len(prog))
    out.write("\n".join(prog))
    out.write("\nout " + " ".join(str(x) for x in out_line) + "\n")


if __name__ == "__main__":
    main()
