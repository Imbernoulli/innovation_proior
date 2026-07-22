# TIER: trivial
# Naive baseline: simulate every AGENT individually and, for every round, sum
# every individual incoming edge one at a time (no community aggregation, no
# collapsing of identical agents). Because the circuit must be correct for
# EVERY per-community bias beta in [1,CAP]^K (beta is an unknown, substituted
# by the checker -- see counter.py), it unrolls the beta-independent
# worst-case round bound R_max = K*CAP + 2. This literally realizes the
# checker's canonical baseline B.
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
    R_max = K * CAP + 2

    prog = []

    def emit_const(v):
        prog.append("const %d" % v)
        return K + len(prog) - 1

    def emit_op(op, i, j):
        prog.append("%s %d %d" % (op, i, j))
        return K + len(prog) - 1

    zero = emit_const(0)
    cur = [zero for _ in range(N)]          # x_i^(0) = 0
    capc = emit_const(CAP)

    for _ in range(R_max):
        new_cur = [0] * N
        for i in range(N):
            a = c[i]
            terms = []
            for j in range(N):
                if j == i:
                    continue
                w = W[a][c[j]]
                if w == 0:
                    continue
                wc = emit_const(w)
                terms.append(emit_op("mul", wc, cur[j]))
            bc = a                            # implicit beta input node for community a
            if terms:
                s = terms[0]
                for t2 in terms[1:]:
                    s = emit_op("add", s, t2)
                s = emit_op("add", s, bc)
            else:
                s = bc
            s = emit_op("min", s, capc)
            new_cur[i] = s
        cur = new_cur

    out = sys.stdout
    out.write("%d\n" % len(prog))
    out.write("\n".join(prog))
    out.write("\nout " + " ".join(str(cur[i]) for i in range(N)) + "\n")


if __name__ == "__main__":
    main()
