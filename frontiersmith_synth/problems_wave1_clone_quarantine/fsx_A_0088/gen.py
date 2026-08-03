import sys, random

# gen.py <testId>  -- prints ONE beamforming gain tensor instance to stdout.
# The tensor is planted as a sum of R_plant rank-1 (separable) stages with small
# nonzero integer weights, so it has genuine low-rank structure BUT the planted
# rank exceeds the largest dimension (overcomplete) -- Jennrich / simultaneous
# diagonalization (which need rank <= dimension) cannot recover the optimum, and
# the true minimal rank stays unknown. Difficulty grows with testId.
#
# Shapes always have c > b so slicing along the last (time) axis is strictly
# worse than the best axis -- this separates the greedy and strong strategies.

# (a, b, c, R_plant)  with a <= b < c, max dim <= 5, R_plant > max dim.
SPECS = {
    1:  (2, 3, 4, 5),
    2:  (2, 3, 5, 6),
    3:  (2, 4, 5, 6),
    4:  (3, 3, 4, 5),
    5:  (3, 3, 5, 6),
    6:  (3, 4, 5, 7),
    7:  (2, 4, 5, 7),
    8:  (3, 4, 5, 8),
    9:  (4, 4, 5, 8),
    10: (4, 4, 5, 9),
}

def main():
    tid = int(sys.argv[1])
    a, b, c, R = SPECS[tid]
    rng = random.Random(918273 + 1000 * tid)

    def vec(n):
        # nonzero small integer weights keep the tensor dense
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
