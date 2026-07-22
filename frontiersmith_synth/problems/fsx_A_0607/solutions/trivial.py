# TIER: trivial
# Canonical naive per-query plan (no sharing): for each query, for each monomial
# form  x_i * x_j  (1 mul), scale by coef (1 mul), accumulate (1 add).  This
# reproduces the checker's baseline op count exactly -> Ratio ~ 0.1.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); K = int(next(it))
    queries = []
    for _ in range(K):
        Tq = int(next(it))
        terms = []
        for _t in range(Tq):
            i = int(next(it)); j = int(next(it)); c = int(next(it))
            terms.append((i, j, c))
        queries.append(terms)

    ins = []      # (op, a, b)

    def emit(op, a, b):
        ins.append((op, a, b))
        return "r%d" % (len(ins) - 1)

    outs = []
    for terms in queries:
        acc = None
        for (i, j, c) in terms:
            prod = emit("*", "x%d" % i, "x%d" % j)
            tm = emit("*", str(c), prod)
            acc = tm if acc is None else emit("+", acc, tm)
        outs.append(acc if acc is not None else "0")

    toks = [str(len(ins))]
    for op, a, b in ins:
        toks += [op, a, b]
    toks += outs
    sys.stdout.write(" ".join(toks) + "\n")


if __name__ == "__main__":
    main()
