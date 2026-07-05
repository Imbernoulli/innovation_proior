# TIER: greedy
# Pick the CHEAPEST of the three FIBER decompositions (fewest non-zero fibers over a mode).
#   mode-3 fibers -> ~I*J terms ; mode-2 -> ~I*K ; mode-1 -> ~J*K.
# No rank reduction is done -- just the least-populated orientation. Beats trivial whenever
# an off-frontal mode has fewer non-zero fibers, but stays far above the slice-rank scheme.
import sys


def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    I = int(next(it)); J = int(next(it)); K = int(next(it))
    T = [[[0] * K for _ in range(J)] for _ in range(I)]
    for k in range(K):
        for i in range(I):
            for j in range(J):
                T[i][j][k] = int(next(it))

    def e(n, x):
        return [1 if p == x else 0 for p in range(n)]

    def mode3():  # fiber over k, indexed by (i,j)
        ts = []
        for i in range(I):
            for j in range(J):
                fib = [T[i][j][k] for k in range(K)]
                if any(v for v in fib):
                    ts.append((e(I, i), e(J, j), fib))
        return ts

    def mode2():  # fiber over j, indexed by (i,k)
        ts = []
        for i in range(I):
            for k in range(K):
                fib = [T[i][j][k] for j in range(J)]
                if any(v for v in fib):
                    ts.append((e(I, i), fib, e(K, k)))
        return ts

    def mode1():  # fiber over i, indexed by (j,k)
        ts = []
        for j in range(J):
            for k in range(K):
                fib = [T[i][j][k] for i in range(I)]
                if any(v for v in fib):
                    ts.append((fib, e(J, j), e(K, k)))
        return ts

    best = min([mode1(), mode2(), mode3()], key=len)

    out = [str(len(best))]
    for a, b, c in best:
        out.append(" ".join(map(str, a)))
        out.append(" ".join(map(str, b)))
        out.append(" ".join(map(str, c)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
