# TIER: greedy
# The obvious recipe: GLOBAL monomial common-subexpression elimination.  Compute
# each distinct product  x_i * x_j  once (shared across all queries), then form
# each query as a linear combination of these products.  Beats the naive plan by
# sharing the low-level products -- but it never discovers the shared LINEAR-FORM
# structure, so it still pays for the full (large) set of expanded monomials.
import sys
from fractions import Fraction


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

    ins = []

    def emit(op, a, b):
        ins.append((op, a, b))
        return "r%d" % (len(ins) - 1)

    def frac(c):
        c = Fraction(c)
        return str(c.numerator) if c.denominator == 1 else "%d/%d" % (c.numerator, c.denominator)

    def lincomb(pairs):
        pairs = [(Fraction(c), o) for c, o in pairs if c != 0]
        if not pairs:
            return "0"

        def scaled(c, o):
            if c == 1:
                return ("+", o)
            if c == -1:
                return ("-", o)
            return ("+", emit("*", frac(c), o))
        s0, first = scaled(*pairs[0])
        acc = first if s0 == "+" else emit("-", "0", first)
        for c, o in pairs[1:]:
            s, r = scaled(c, o)
            acc = emit("+" if s == "+" else "-", acc, r)
        return acc

    # distinct monomials across ALL queries -> one shared product each
    monos = sorted({(i, j) for terms in queries for (i, j, c) in terms})
    preg = {}
    for (i, j) in monos:
        preg[(i, j)] = emit("*", "x%d" % i, "x%d" % j)

    outs = []
    for terms in queries:
        outs.append(lincomb([(c, preg[(i, j)]) for (i, j, c) in terms]))

    toks = [str(len(ins))]
    for op, a, b in ins:
        toks += [op, a, b]
    toks += outs
    sys.stdout.write(" ".join(toks) + "\n")


if __name__ == "__main__":
    main()
