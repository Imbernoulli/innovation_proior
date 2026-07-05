import sys, random

# gen.py <testId>  -- prints ONE quay-transfer tensor instance to stdout.
#
# Harbor skin: T[i][j][k] is the net number of container moves that crane
# program couples vessel-berth i, yard-block j and time-window k.  We plant the
# tensor as a sum of R_plant separable "crane stages" u (x) v (x) w with small
# nonzero integer weights, so the tensor has genuine low-rank structure -- BUT
# the planted rank exceeds every dimension (it is OVERCOMPLETE).  Algebraic
# recovery methods (Jennrich / simultaneous diagonalization) require rank <=
# dimension, so they cannot recover a minimal program; the true minimal number
# of scalar multiplications stays unknown.  Difficulty grows with testId.
#
# The axis lengths deliberately VARY in which one is smallest, so slicing along
# a single fixed axis (the greedy strategy, axis 0 here) is data-dependently
# suboptimal versus the best-of-three-axes strong strategy.

# testId -> (I, J, K, R_plant),  every dim in [2,5], R_plant > max(I,J,K).
SPECS = {
    1:  (4, 2, 3, 5),
    2:  (5, 2, 3, 6),
    3:  (4, 3, 2, 6),
    4:  (5, 3, 2, 6),
    5:  (3, 4, 2, 6),
    6:  (5, 3, 4, 7),
    7:  (4, 3, 5, 7),
    8:  (5, 4, 3, 8),
    9:  (4, 5, 3, 8),
    10: (5, 4, 2, 9),
}


def main():
    tid = int(sys.argv[1])
    I, J, K, R = SPECS[tid]
    rng = random.Random(20260702 + 7919 * tid)

    def vec(n):
        # nonzero small integer weights keep every slab dense & generically full rank
        return [rng.choice([-2, -1, 1, 2]) for _ in range(n)]

    T = [[[0] * K for _ in range(J)] for _ in range(I)]
    for _ in range(R):
        u = vec(I); v = vec(J); w = vec(K)
        for i in range(I):
            for j in range(J):
                uv = u[i] * v[j]
                if uv == 0:
                    continue
                for k in range(K):
                    T[i][j][k] += uv * w[k]

    # safety: never emit an all-zero tensor
    if all(T[i][j][k] == 0 for i in range(I) for j in range(J) for k in range(K)):
        T[0][0][0] = 1

    out = ["%d %d %d" % (I, J, K)]
    for i in range(I):
        for j in range(J):
            out.append(" ".join(str(T[i][j][k]) for k in range(K)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
