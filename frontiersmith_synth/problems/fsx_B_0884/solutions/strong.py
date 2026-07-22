# TIER: strong
# The insight: x_i^(t) depends on i ONLY through its community c_i (an exact
# automorphism of the all-zero start state under the community-symmetric
# update), so the whole N-agent dynamical system is really a K-dimensional
# one. Solve the K-dimensional fixed point ONCE -- with the correct
# multiplicities (n[b] cross-community neighbors, or n[a]-1 same-community
# neighbors, baked directly into a single constant coefficient, no runtime
# subtraction needed) -- and then answer every one of the N agents by
# pointing at its community's representative node. That final broadcast is
# FREE (the `out` line just repeats node indices), so total cost is
# independent of N: it scales with the number of nonzero community links,
# not the number of agents. Because beta is unknown at circuit-build time
# (substituted by the checker), the circuit unrolls the beta-independent
# worst-case round bound R_max = K*CAP + 2 -- still with only K nodes/round.
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
    yn = [zero for _ in range(K)]   # y_a^(0) = 0, ONE node per community
    capc = emit_const(CAP)

    for _ in range(R_max):
        new_yn = [0] * K
        for a in range(K):
            terms = []
            for b in nz[a]:
                coef = W[a][b] * mult(a, b, n)     # multiplicity baked into ONE constant
                cc = emit_const(coef)
                terms.append(emit_op("mul", cc, yn[b]))
            bc = a                                  # implicit beta input node for community a
            if terms:
                s = terms[0]
                for t2 in terms[1:]:
                    s = emit_op("add", s, t2)
                s = emit_op("add", s, bc)
            else:
                s = bc
            s = emit_op("min", s, capc)
            new_yn[a] = s
        yn = new_yn

    out_line = [yn[c[i]] for i in range(N)]   # free broadcast: repeated indices allowed

    out = sys.stdout
    out.write("%d\n" % len(prog))
    out.write("\n".join(prog))
    out.write("\nout " + " ".join(str(x) for x in out_line) + "\n")


if __name__ == "__main__":
    main()
