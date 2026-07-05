import sys, random
from fractions import Fraction

# gen.py <testId>  -- prints ONE abyssal-crosstalk coupling tensor to stdout.
#
# Deep-sea-cable skin: a repeater sits between a bundles of INbound fibres,
# b OUTbound fibres and c wavelength (DWDM) channels.  The measured optical
# crosstalk is a 3-way tensor  T[i][j][k]  (inbound i -> outbound j on channel k).
# A repeater "mixing network" realises T as a sum of separable rank-1 stages;
# each stage costs one scalar multiplier, so minimising the CP rank R minimises
# the multiplier count of the amplifier board.
#
# The tensor is PLANTED as a sum of R_plant rank-1 stages with small nonzero
# integer weights (dense, integer target).  Crucially R_plant > max(a,b,c)
# (OVERCOMPLETE), so Jennrich / simultaneous-diagonalisation -- which need
# rank <= dimension -- cannot recover the planted factors, and the true minimal
# rank stays unknown.  Shapes obey a <= b < c so slicing along the last
# (wavelength) axis is strictly worse than the best axis: greedy != strong.
#
# gen guarantees, by deterministic reseeding, that the emitted tensor is fully
# dense (every coefficient nonzero) and that every axis' slices are full rank,
# so the reference ladder hits its intended operation counts.

# (a, b, c, R_plant) with a <= b < c <= 8 and R_plant > c.
SPECS = {
    1:  (3, 4, 5, 6),
    2:  (3, 4, 6, 7),
    3:  (3, 5, 6, 7),
    4:  (4, 5, 6, 7),
    5:  (3, 5, 7, 8),
    6:  (4, 5, 7, 8),
    7:  (4, 6, 7, 8),
    8:  (4, 5, 8, 9),
    9:  (5, 6, 8, 9),
    10: (4, 6, 8, 9),
}


def mat_rank(M, nr, nc):
    """Exact integer/rational rank via fraction elimination."""
    A = [[Fraction(x) for x in row] for row in M]
    r = 0
    for col in range(nc):
        piv = None
        for rr in range(r, nr):
            if A[rr][col] != 0:
                piv = rr
                break
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        inv = Fraction(1) / A[r][col]
        A[r] = [x * inv for x in A[r]]
        for rr in range(nr):
            if rr != r and A[rr][col] != 0:
                f = A[rr][col]
                A[rr] = [x - f * y for x, y in zip(A[rr], A[r])]
        r += 1
        if r == nr:
            break
    return r


def build(a, b, c, R, seed):
    rng = random.Random(seed)

    def vec(n):
        return [rng.choice([-2, -1, 1, 2]) for _ in range(n)]

    T = [[[0] * c for _ in range(b)] for _ in range(a)]
    for _ in range(R):
        u = vec(a); v = vec(b); w = vec(c)
        for i in range(a):
            for j in range(b):
                uv = u[i] * v[j]
                for k in range(c):
                    T[i][j][k] += uv * w[k]
    return T


def full_rank_all_axes(T, a, b, c):
    # axis 0 slices: b x c ; axis 1 slices: a x c ; axis 2 slices: a x b
    for i in range(a):
        if mat_rank([[T[i][j][k] for k in range(c)] for j in range(b)], b, c) != min(b, c):
            return False
    for j in range(b):
        if mat_rank([[T[i][j][k] for k in range(c)] for i in range(a)], a, c) != min(a, c):
            return False
    for k in range(c):
        if mat_rank([[T[i][j][k] for j in range(b)] for i in range(a)], a, b) != min(a, b):
            return False
    return True


def dense(T, a, b, c):
    return all(T[i][j][k] != 0 for i in range(a) for j in range(b) for k in range(c))


def main():
    tid = int(sys.argv[1])
    a, b, c, R = SPECS[tid]
    base = 550_000 + 10_000 * tid
    T = None
    for s in range(base, base + 200000):
        cand = build(a, b, c, R, s)
        if dense(cand, a, b, c) and full_rank_all_axes(cand, a, b, c):
            T = cand
            break
    if T is None:                       # deterministic fallback (never expected)
        T = build(a, b, c, R, base)

    out = ["%d %d %d" % (a, b, c)]
    for i in range(a):
        for j in range(b):
            out.append(" ".join(str(T[i][j][k]) for k in range(c)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
