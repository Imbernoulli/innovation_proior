# TIER: greedy
# The "obvious strong-coder" optimization: precompute a per-community sum
# once per round (common-subexpression elimination across agents that share
# a neighbor community) so each agent's own update only costs O(#nonzero
# community links) instead of O(#neighbor agents). Still stops short of
# noticing every agent inside a community carries the exact same value --
# it keeps re-deriving N separate per-agent nodes every round instead of
# collapsing to K representatives. Like trivial, it must stay correct for
# every beta in [1,CAP]^K, so it unrolls R_max = K*CAP + 2 rounds.
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
    groups = [[] for _ in range(K)]
    for i, ci in enumerate(c):
        n[ci] += 1
        groups[ci].append(i)
    R_max = K * CAP + 2

    prog = []

    def emit_const(v):
        prog.append("const %d" % v)
        return K + len(prog) - 1

    def emit_op(op, i, j):
        prog.append("%s %d %d" % (op, i, j))
        return K + len(prog) - 1

    zero = emit_const(0)
    cur = [zero for _ in range(N)]           # x_i^(0) = 0
    capc = emit_const(CAP)

    for _ in range(R_max):
        # per-community aggregate sums (one shared subexpression per community)
        S = [None] * K
        for b in range(K):
            members = groups[b]
            if not members:
                continue
            s = cur[members[0]]
            for j2 in members[1:]:
                s = emit_op("add", s, cur[j2])
            S[b] = s

        new_cur = [0] * N
        for i in range(N):
            a = c[i]
            terms = []
            for b in range(K):
                w = W[a][b]
                if w == 0 or S[b] is None:
                    continue
                if b == a:
                    if n[a] <= 1:
                        continue
                    selfsum = emit_op("sub", S[a], cur[i])
                    wc = emit_const(w)
                    terms.append(emit_op("mul", wc, selfsum))
                else:
                    wc = emit_const(w)
                    terms.append(emit_op("mul", wc, S[b]))
            bc = a                             # implicit beta input node for community a
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
