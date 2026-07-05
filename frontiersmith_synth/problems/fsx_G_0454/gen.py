import sys, random

# gen.py <testId>  -- prints ONE integer target matrix M (n rows, m cols) to stdout.
#
# M is PLANTED as a sum of R_plant rank-1 outer products a_k b_k^T whose factor
# entries live in the TERNARY alphabet {-1,0,1}.  So M is exactly representable by
# R_plant ternary rank-1 terms (an upper bound on the answer), and its real rank is
# a lower bound.  The plant is OVERCOMPLETE (R_plant can exceed the real rank because
# the ternary factors are linearly dependent), so neither bound is tight: the minimal
# number of ternary rank-1 terms ("ternary rank") is NP-hard and its exact value is
# unknown.  Difficulty (n, m, R_plant, density) grows with testId.
#
# Shapes always use m < n so a column-wise construction (m terms per level) is
# cheaper than the row-wise baseline (n terms per level) -- this separates the
# greedy strategy from the trivial one.

# (n_rows, m_cols, R_plant, p_nonzero)
SPECS = {
    1:  (8,  5,  3, 0.55),
    2:  (10, 6,  3, 0.55),
    3:  (11, 6,  4, 0.55),
    4:  (12, 7,  4, 0.50),
    5:  (13, 7,  5, 0.50),
    6:  (14, 8,  5, 0.50),
    7:  (16, 9,  6, 0.48),
    8:  (18, 10, 6, 0.48),
    9:  (20, 11, 7, 0.46),
    10: (22, 12, 8, 0.46),
}

def main():
    tid = int(sys.argv[1])
    n, m, R, p = SPECS[tid]
    rng = random.Random(730451 + 1000 * tid)

    def tern(k):
        out = []
        for _ in range(k):
            if rng.random() < p:
                out.append(rng.choice([-1, 1]))
            else:
                out.append(0)
        return out

    M = [[0] * m for _ in range(n)]
    nonzero = False
    tries = 0
    while not nonzero and tries < 100:
        M = [[0] * m for _ in range(n)]
        for _ in range(R):
            a = tern(n); b = tern(m)
            for i in range(n):
                if a[i] == 0:
                    continue
                for j in range(m):
                    if b[j]:
                        M[i][j] += a[i] * b[j]
        nonzero = any(M[i][j] != 0 for i in range(n) for j in range(m))
        tries += 1

    out = ["%d %d" % (n, m)]
    for i in range(n):
        out.append(" ".join(str(M[i][j]) for j in range(m)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
