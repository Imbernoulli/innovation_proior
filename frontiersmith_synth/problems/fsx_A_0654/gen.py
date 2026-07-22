import sys, random

# gen.py <testId>  -- prints ONE greenhouse-ledger instance to stdout.
#
# S storage compartments evolve for T days under an affine update
#   x_{t+1} = A_t * x_t + b_t      (all arithmetic taken mod MOD at read/answer time)
# The per-day map (A_t,b_t) is chosen from K "recipe" blocks that ROTATE with a
# planted period p (pattern[t % p] picks the block), EXCEPT on m irregular
# "recalibration" days where a one-off dense graft map overrides the rotation.
# Each recipe block A_k is BANDED (only |i-j|<=W nonzero) -- a real neighbor-only
# coupling; graft matrices are dense (no structure) so they cannot be skipped or
# folded into the rotation. Difficulty (T, p, m) grows with testId; the last few
# tests have long banded/periodic runs between recalibration days -- the trap
# that separates step-by-step simulation from recognizing the algebraic
# (period-composition) structure.

MOD = 1_000_000_007
S = 6
W = 1  # bandwidth: A_k[i][j] != 0 only if |i-j| <= W

# (T, p, m) ladder, difficulty increasing with testId.
LADDER = {
    1:  (5,   3, 1),
    2:  (9,   3, 1),
    3:  (16,  4, 1),
    4:  (30,  4, 1),
    5:  (60,  5, 1),
    6:  (150, 5, 2),
    7:  (350, 6, 2),
    8:  (800, 6, 2),
    9:  (1600, 7, 3),
    10: (3000, 7, 7),
}
K = 3  # number of rotating recipe blocks


def rnd_val(rng):
    return rng.randrange(1, 1000)


def banded_matrix(rng):
    A = [[0] * S for _ in range(S)]
    for i in range(S):
        for j in range(S):
            if abs(i - j) <= W:
                A[i][j] = rnd_val(rng)
    return A


def dense_matrix(rng):
    return [[rnd_val(rng) for _ in range(S)] for _ in range(S)]


def vec(rng):
    return [rnd_val(rng) for _ in range(S)]


def main():
    tid = int(sys.argv[1])
    T, p, m = LADDER[tid]
    rng = random.Random(20260710 + 97 * tid)

    x0 = vec(rng)

    A = [banded_matrix(rng) for _ in range(K)]
    b = [vec(rng) for _ in range(K)]

    pattern = [rng.randrange(K) for _ in range(p)]

    m = min(m, max(0, T - 1))
    positions = sorted(rng.sample(range(T), m)) if m > 0 else []
    grafts = []
    for t in positions:
        G = dense_matrix(rng)
        h = vec(rng)
        grafts.append((t, G, h))

    out = []
    out.append("%d %d %d %d %d" % (S, K, T, p, m))
    out.append(" ".join(map(str, x0)))
    for k in range(K):
        for row in A[k]:
            out.append(" ".join(map(str, row)))
        out.append(" ".join(map(str, b[k])))
    out.append(" ".join(map(str, pattern)))
    for (t, G, h) in grafts:
        out.append(str(t))
        for row in G:
            out.append(" ".join(map(str, row)))
        out.append(" ".join(map(str, h)))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
