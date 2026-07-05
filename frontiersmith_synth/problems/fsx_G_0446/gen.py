#!/usr/bin/env python3
# gen.py <testId> : print ONE instance of the planted-overcomplete bilinear-form
# multiplication-minimization problem. testId 1..10 = difficulty ladder.
# All randomness is seeded ONLY by testId -> bit-for-bit reproducible.
import sys, random

# (m=n=M, d = per-slice inner dim, rho = # independent slice patterns, P = # slices)
CONFIGS = [
    (4, 2, 2, 4), (5, 2, 3, 5), (5, 3, 2, 4), (6, 2, 3, 5), (6, 3, 3, 5),
    (6, 3, 4, 6), (7, 3, 3, 5), (7, 3, 4, 6), (7, 4, 3, 5), (8, 4, 4, 6),
]

def matmul(A, B):
    m = len(A); k = len(A[0]); n = len(B[0])
    return [[sum(A[i][t] * B[t][j] for t in range(k)) for j in range(n)] for i in range(m)]

def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr); sys.exit(2)
    t = int(sys.argv[1])
    if t < 1 or t > len(CONFIGS):
        t = ((t - 1) % len(CONFIGS)) + 1
    M, d, rho, P = CONFIGS[t - 1]
    rng = random.Random(20260701 * 131 + t)          # seeded by testId only
    def ri(): return rng.randint(-3, 3)

    # shared left / right pools (columns span a d-dim subspace of R^M)
    U = [[ri() for _ in range(d)] for _ in range(M)]          # M x d
    Vt = [[ri() for _ in range(M)] for _ in range(d)]         # d x M  ( = V^T )
    # rho independent core patterns -> rho independent base slices S_k = U A_k V^T
    bases = []
    for _ in range(rho):
        A = [[ri() for _ in range(d)] for _ in range(d)]      # d x d
        bases.append(matmul(matmul(U, A), Vt))               # M x M
    # P slices: first rho are the base slices, the rest are integer combos of them
    slices = [bases[k] for k in range(rho)]
    for _ in range(rho, P):
        coeffs = [rng.randint(-2, 2) for _ in range(rho)]
        if all(c == 0 for c in coeffs):
            coeffs[0] = 1
        slices.append([[sum(coeffs[k] * bases[k][i][j] for k in range(rho))
                        for j in range(M)] for i in range(M)])

    out = [f"{P} {M} {M}"]
    for S in slices:
        for row in S:
            out.append(" ".join(str(x) for x in row))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
