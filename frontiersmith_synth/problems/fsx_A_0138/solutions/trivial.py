# TIER: trivial
# Schoolbook CP decomposition: one rank-1 term per non-zero entry.
# R = nnz(T) -> exactly reproduces the checker baseline -> ratio ~= 0.1.
import sys


def main():
    inp = sys.stdin.read().split()
    I, J, K = int(inp[0]), int(inp[1]), int(inp[2])
    idx = 3
    T = [[[0] * K for _ in range(J)] for _ in range(I)]
    for i in range(I):
        for j in range(J):
            for k in range(K):
                T[i][j][k] = int(inp[idx]); idx += 1

    terms = []
    for i in range(I):
        for j in range(J):
            for k in range(K):
                v = T[i][j][k]
                if v == 0:
                    continue
                a = [0] * I; a[i] = v
                b = [0] * J; b[j] = 1
                c = [0] * K; c[k] = 1
                terms.append((a, b, c))

    out = [str(len(terms))]
    for a, b, c in terms:
        out.append(" ".join(map(str, a)))
        out.append(" ".join(map(str, b)))
        out.append(" ".join(map(str, c)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
