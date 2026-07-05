# TIER: greedy
# Pick the cheapest of the three flattenings (put the two unit-vectors on the
# mode-pair with the fewest nonzero fibres; the remaining mode carries the fibre).
import sys

def read_tensor():
    tok = sys.stdin.read().split()
    it = iter(tok)
    a = int(next(it)); b = int(next(it)); c = int(next(it))
    T = [[[0] * c for _ in range(b)] for _ in range(a)]
    for k in range(c):
        for i in range(a):
            for j in range(b):
                T[i][j][k] = int(next(it))
    return a, b, c, T

def main():
    a, b, c, T = read_tensor()

    def build_c():  # unit on (i,j), fibre over k
        terms = []
        for i in range(a):
            for j in range(b):
                fib = [T[i][j][k] for k in range(c)]
                if any(x != 0 for x in fib):
                    terms.append(([1 if r==i else 0 for r in range(a)],
                                  [1 if r==j else 0 for r in range(b)], fib))
        return terms

    def build_a():  # unit on (j,k), fibre over i
        terms = []
        for j in range(b):
            for k in range(c):
                fib = [T[i][j][k] for i in range(a)]
                if any(x != 0 for x in fib):
                    terms.append((fib,
                                  [1 if r==j else 0 for r in range(b)],
                                  [1 if r==k else 0 for r in range(c)]))
        return terms

    def build_b():  # unit on (i,k), fibre over j
        terms = []
        for i in range(a):
            for k in range(c):
                fib = [T[i][j][k] for j in range(b)]
                if any(x != 0 for x in fib):
                    terms.append(([1 if r==i else 0 for r in range(a)],
                                  fib,
                                  [1 if r==k else 0 for r in range(c)]))
        return terms

    cand = [build_c(), build_a(), build_b()]
    terms = min(cand, key=len)

    out = [str(len(terms))]
    for (u, v, w) in terms:
        out.append(" ".join(map(str, u)))
        out.append(" ".join(map(str, v)))
        out.append(" ".join(map(str, w)))
    sys.stdout.write("\n".join(out) + "\n")

main()
