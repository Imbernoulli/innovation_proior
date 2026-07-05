#!/usr/bin/env python3
"""gen.py <testId>  -- print ONE instance of the thermal-coupling tensor problem.

An instance is a dense I x J x K integer tensor T of thermal-coupling
coefficients, built as a sum of `r` positive rank-1 "cooling modes" (so every
entry is strictly positive => the tensor is fully dense).  `r` is chosen larger
than every dimension (an OVERCOMPLETE planted rank) so the planted factors are
NOT recoverable by any known polynomial algorithm (Jennrich / simultaneous
diagonalisation require rank <= dimension).  The true tensor rank is therefore
genuinely unknown, making the objective open-ended.

Difficulty ladder (testId 1..10): the dimensions grow.  Everything is seeded by
testId only, so generation is bit-for-bit reproducible.
"""
import sys, random

# (I, J, K) with I <= J < K.  K <= 8 keeps the strongest reference well below
# score saturation, leaving head-room above it.
LADDER = [
    (2, 3, 4),
    (2, 4, 5),
    (3, 4, 5),
    (3, 4, 6),
    (3, 5, 6),
    (4, 5, 6),
    (4, 5, 7),
    (4, 6, 7),
    (5, 6, 7),
    (5, 6, 8),
]


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    t = int(sys.argv[1])
    I, J, K = LADDER[(t - 1) % len(LADDER)]
    rng = random.Random(1000 + t)          # seed depends on testId ONLY

    r = max(I, J, K) + 2                    # overcomplete planted rank
    # positive integer factor matrices -> tensor is strictly positive everywhere
    A = [[rng.randint(1, 4) for _ in range(I)] for _ in range(r)]
    B = [[rng.randint(1, 4) for _ in range(J)] for _ in range(r)]
    C = [[rng.randint(1, 4) for _ in range(K)] for _ in range(r)]

    T = [[[0] * K for _ in range(J)] for _ in range(I)]
    for s in range(r):
        a, b, c = A[s], B[s], C[s]
        for i in range(I):
            ai = a[i]
            for j in range(J):
                aibj = ai * b[j]
                row = T[i][j]
                for k in range(K):
                    row[k] += aibj * c[k]

    out = [f"{I} {J} {K}"]
    for i in range(I):
        for j in range(J):
            out.append(" ".join(str(x) for x in T[i][j]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
