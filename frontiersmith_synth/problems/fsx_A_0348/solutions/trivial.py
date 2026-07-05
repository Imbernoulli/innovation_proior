# TIER: trivial
# Fiber decomposition along mode 3: one rank-1 term per (i,j) cell.
#   T = sum_{i,j} e_i (x) e_j (x) T[i][j][:]     ->  R = I*J  (== checker baseline B -> ratio 0.1)
import sys


def main():
    toks = sys.stdin.read().split()
    idx = 0
    I = int(toks[idx]); idx += 1
    J = int(toks[idx]); idx += 1
    K = int(toks[idx]); idx += 1
    T = [[[0] * K for _ in range(J)] for _ in range(I)]
    for i in range(I):
        for j in range(J):
            for k in range(K):
                T[i][j][k] = int(toks[idx]); idx += 1

    out = []
    terms = []
    for i in range(I):
        for j in range(J):
            a = [0] * I; a[i] = 1
            b = [0] * J; b[j] = 1
            c = T[i][j][:]
            terms.append((a, b, c))
    out.append(str(len(terms)))
    for (a, b, c) in terms:
        out.append(" ".join(str(x) for x in (a + b + c)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
