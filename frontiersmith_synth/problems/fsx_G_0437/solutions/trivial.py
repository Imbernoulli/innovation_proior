# TIER: trivial
# Mode-c flattening: one rank-1 term per nonzero (i,j) fibre.
# term = e_i (x) e_j (x) fibre_over_k .  R = # nonzero fibres = baseline B.
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
    terms = []
    for i in range(a):
        for j in range(b):
            fib = [T[i][j][k] for k in range(c)]
            if any(x != 0 for x in fib):
                u = [1 if r == i else 0 for r in range(a)]
                v = [1 if r == j else 0 for r in range(b)]
                terms.append((u, v, fib))
    out = [str(len(terms))]
    for (u, v, w) in terms:
        out.append(" ".join(map(str, u)))
        out.append(" ".join(map(str, v)))
        out.append(" ".join(map(str, w)))
    sys.stdout.write("\n".join(out) + "\n")

main()
