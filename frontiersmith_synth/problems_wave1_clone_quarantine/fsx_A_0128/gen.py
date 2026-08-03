import sys, random

# Tide-pool interaction-tensor generator.
# Prints a dense integer 3D tensor T of shape (a,b,c) that is guaranteed to be
# expressible as a sum of R0 rank-1 "interaction channels", but is presented as
# a raw tensor. Finding a short channel list (low CP rank) is the challenge.
#
# Output format:
#   line 1: a b c
#   then, for k = 0..c-1 (each tidal-state slice), a lines of b integers
#          giving T[i][j][k]  (species i, nutrient j).

# difficulty ladder: (a, b, c, R0)
LADDER = [
    (3, 3, 3, 2),
    (3, 3, 4, 3),
    (3, 4, 4, 3),
    (3, 4, 5, 4),
    (4, 4, 4, 4),
    (4, 4, 5, 4),
    (4, 5, 5, 5),
    (4, 5, 6, 5),
    (5, 5, 5, 5),
    (5, 5, 6, 6),
]

def build(a, b, c, R0, rng):
    while True:
        X = [[rng.randint(-2, 2) for _ in range(R0)] for _ in range(a)]
        Y = [[rng.randint(-2, 2) for _ in range(R0)] for _ in range(b)]
        Z = [[rng.randint(-2, 2) for _ in range(R0)] for _ in range(c)]
        T = [[[0] * c for _ in range(b)] for _ in range(a)]
        for r in range(R0):
            for i in range(a):
                for j in range(b):
                    for k in range(c):
                        T[i][j][k] += X[i][r] * Y[j][r] * Z[k][r]
        nnz = sum(1 for i in range(a) for j in range(b) for k in range(c) if T[i][j][k] != 0)
        total = a * b * c
        # require a dense-ish, non-trivial tensor
        if nnz >= max(1, int(0.75 * total)) and any(
            T[i][j][k] != 0 for i in range(a) for j in range(b) for k in range(c)):
            return T

def main():
    tid = int(sys.argv[1])
    idx = min(max(tid, 1), len(LADDER)) - 1
    a, b, c, R0 = LADDER[idx]
    rng = random.Random(90128 + 7919 * tid)
    T = build(a, b, c, R0, rng)
    out = ["%d %d %d" % (a, b, c)]
    for k in range(c):
        for i in range(a):
            out.append(" ".join(str(T[i][j][k]) for j in range(b)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
