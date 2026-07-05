import sys, random

# gen.py <testId>  -- prints ONE generalized short-convolution accelerator instance.
#
# Winograd-style short-convolution framing: a signal tile a (length p) is combined
# with a filter tile b (length q) to produce s pooled output taps.  The accelerator
# is a fixed BILINEAR map
#         c[k] = sum_{i,j} T[k][i][j] * a[i] * b[j]      (k = 0..s-1)
# and the design goal is to realise it with as few scalar MULTIPLIERS as possible
# (a Winograd/bilinear algorithm: rank of the structure tensor T).
#
# T is PLANTED as a sum of R_plant rank-1 bilinear terms with small nonzero integer
# weights.  R_plant is chosen OVERCOMPLETE (R_plant > every mode dimension) so that
# spectral recovery methods (Jennrich / simultaneous diagonalisation, which require
# rank <= dimension) cannot reconstruct the minimal algorithm -- the true minimal
# multiplier count stays UNKNOWN and below any easy mode-slicing bound.
#
# The output pooling dimension s is kept SMALL (s < q < p) so that sharing a product
# across output taps genuinely pays, while the signal dimension p is the largest so
# that slicing along the signal axis (the naive "greedy" choice) is strictly wasteful.

# (p, q, s, R_plant) with s < q < p and R_plant > p ; difficulty grows with testId.
SPECS = {
    1:  (5, 3, 2, 7),
    2:  (5, 4, 2, 8),
    3:  (6, 4, 2, 8),
    4:  (6, 5, 3, 9),
    5:  (7, 5, 3, 10),
    6:  (7, 6, 3, 11),
    7:  (8, 6, 3, 12),
    8:  (8, 7, 4, 13),
    9:  (9, 7, 4, 14),
    10: (9, 8, 4, 15),
}

def main():
    tid = int(sys.argv[1])
    p, q, s, R = SPECS[tid]
    rng = random.Random(918273 + 1000 * tid)

    def vec(n):
        # nonzero small integer weights keep every product fiber dense
        return [rng.choice([-2, -1, 1, 2]) for _ in range(n)]

    # T[k][i][j] : output tap k, signal sample i, filter tap j
    T = [[[0] * q for _ in range(p)] for _ in range(s)]
    for _ in range(R):
        u = vec(p); v = vec(q); w = vec(s)
        for k in range(s):
            for i in range(p):
                wu = w[k] * u[i]
                for j in range(q):
                    T[k][i][j] += wu * v[j]

    # Layout: header "p q s", then p*q lines.  Line at (i*q + j) lists the output
    # fiber  T[0][i][j] T[1][i][j] ... T[s-1][i][j]  for the product a[i]*b[j].
    out = ["%d %d %d" % (p, q, s)]
    for i in range(p):
        for j in range(q):
            out.append(" ".join(str(T[k][i][j]) for k in range(s)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
