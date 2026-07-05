import sys, random

# gen.py <testId>  -- prints ONE "traffic-signal response tensor" instance to stdout.
#
# THEME (skin): a city intersection has a 3D actuation-response table
#   R[i][j][k]  over  (approach direction i, movement type j, signal phase k).
# Computing every combined actuation naively costs one multiply per nonzero
# coefficient.  A rank-R "shared-gain" decomposition realizes the SAME table with
# R separable multiplier units and is the object to be minimized.
#
# MATH: the tensor is PLANTED as a sum of R_plant rank-1 separable stages with
# small nonzero integer weights, so it has genuine low-rank structure BUT the
# planted rank exceeds the largest dimension (overcomplete).  Jennrich /
# simultaneous diagonalization (rank <= dimension) therefore cannot recover the
# optimum and the true minimal rank stays UNKNOWN -- the ceiling is open.
#
# All shapes satisfy  a <= b < c  (last/phase axis strictly largest) so slicing
# along the fixed last axis is strictly worse than the best axis; this separates
# the greedy (fixed-axis) and strong (best-of-three) strategies.

# (a, b, c, R_plant)  with a <= b < c, max dim <= 5, R_plant > max dim.
SPECS = {
    1:  (2, 3, 4, 6),
    2:  (2, 3, 5, 7),
    3:  (3, 3, 4, 6),
    4:  (2, 4, 5, 7),
    5:  (3, 3, 5, 7),
    6:  (3, 4, 5, 7),
    7:  (2, 4, 5, 8),
    8:  (3, 4, 5, 8),
    9:  (4, 4, 5, 8),
    10: (4, 4, 5, 9),
}


def main():
    tid = int(sys.argv[1])
    if tid not in SPECS:
        tid = ((tid - 1) % 10) + 1
    a, b, c, R = SPECS[tid]
    rng = random.Random(424242 + 1000 * tid)

    def vec(n):
        # nonzero small integer weights keep the tensor dense and slices full-rank
        return [rng.choice([-2, -1, 1, 2]) for _ in range(n)]

    T = [[[0] * c for _ in range(b)] for _ in range(a)]
    for _ in range(R):
        u = vec(a); v = vec(b); w = vec(c)
        for i in range(a):
            for j in range(b):
                uv = u[i] * v[j]
                for k in range(c):
                    T[i][j][k] += uv * w[k]

    out = ["%d %d %d" % (a, b, c)]
    for i in range(a):
        for j in range(b):
            out.append(" ".join(str(T[i][j][k]) for k in range(c)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
