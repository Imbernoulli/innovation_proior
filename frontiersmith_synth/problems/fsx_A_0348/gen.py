#!/usr/bin/env python3
"""
fsx_A_0348  --  gen.py <testId>   (testId 1..10)

Prints ONE instance of the "museum gallery interaction tensor" to stdout.

The instance is a small 3-D integer tensor  T[i][j][k]  with dimensions I x J x K
(2 <= dim <= 5, medium scale).  It is built by PLANTING  R0  rank-1 terms

        T = sum_{r=1..R0}  a_r (x) b_r (x) c_r ,

with small integer factor vectors seeded ONLY by testId.  R0 is chosen
OVERCOMPLETE:  R0 > max(I,J,K)  (so simultaneous-diagonalisation / Jennrich,
which needs rank <= dimension, cannot recover the planted decomposition), while
R0 < the single-mode slice-rank ceiling (so a genuinely better decomposition
than any greedy slice method is known to exist but is hard to find).

Only the tensor T is emitted -- the planted factors are NOT revealed.
"""
import sys, random

# testId -> (I, J, K, R0).   R0 satisfies  max(I,J,K) < R0 < slice-sum ceiling.
DIMS = {
    1:  (4, 3, 2, 5),
    2:  (5, 4, 2, 6),
    3:  (5, 4, 3, 8),
    4:  (4, 4, 3, 7),
    5:  (5, 5, 2, 7),
    6:  (5, 4, 4, 8),
    7:  (4, 3, 3, 6),
    8:  (5, 5, 3, 8),
    9:  (5, 2, 4, 6),
    10: (5, 4, 2, 7),
}


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if t not in DIMS:
        t = ((t - 1) % len(DIMS)) + 1
    I, J, K, R0 = DIMS[t]
    rnd = random.Random(600013 + 977 * t)

    def vec(n):
        return [rnd.choice([-2, -1, 1, 2]) for _ in range(n)]

    T = [[[0] * K for _ in range(J)] for _ in range(I)]
    for _ in range(R0):
        a, b, c = vec(I), vec(J), vec(K)
        for i in range(I):
            ai = a[i]
            for j in range(J):
                aibj = ai * b[j]
                Tij = T[i][j]
                for k in range(K):
                    Tij[k] += aibj * c[k]

    out = ["%d %d %d" % (I, J, K)]
    for i in range(I):
        for j in range(J):
            out.append(" ".join(str(x) for x in T[i][j]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
