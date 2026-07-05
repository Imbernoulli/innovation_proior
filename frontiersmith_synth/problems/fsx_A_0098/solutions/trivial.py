# TIER: trivial
# One separable survey mode per nonzero tensor entry (rank = number of nonzeros).
# Reproduces the checker's baseline -> Ratio ~ 0.1.
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    I = int(next(it)); J = int(next(it)); K = int(next(it))
    T = [[[0] * K for _ in range(J)] for _ in range(I)]
    for i in range(I):
        for j in range(J):
            for k in range(K):
                T[i][j][k] = int(next(it))

    terms = []
    for i in range(I):
        for j in range(J):
            for k in range(K):
                v = T[i][j][k]
                if v != 0:
                    a = [0] * I; a[i] = v
                    b = [0] * J; b[j] = 1
                    c = [0] * K; c[k] = 1
                    terms.append((a, b, c))

    out = [str(len(terms))]
    for (a, b, c) in terms:
        out.append(" ".join(map(str, a + b + c)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
