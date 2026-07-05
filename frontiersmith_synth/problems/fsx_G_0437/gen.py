import sys, random

# Difficulty ladder: build a 3D tensor T of shape (a,b,c) whose mode-c slices
# are each planted rank-2 (over the rationals), while the overall CP rank is
# OVERCOMPLETE (> max dimension) with generic factors -> the true rank stays
# unknown. Entries are integers. Instance printed slice-by-slice.
#
# testId t (1..10): a = b = 4 + t, c = 3 + t. Each mode-c slice M_k = p_k q_k^T
# + s_k t_k^T with integer vectors, guaranteed rank exactly 2 and no zero
# (i,j)->k fibre.

def rank2_int(rows):
    # exact rank of a small integer matrix via fraction elimination
    from fractions import Fraction
    M = [[Fraction(x) for x in r] for r in rows]
    R = len(M); C = len(M[0]) if R else 0
    rank = 0
    for col in range(C):
        piv = None
        for r in range(rank, R):
            if M[r][col] != 0:
                piv = r; break
        if piv is None:
            continue
        M[rank], M[piv] = M[piv], M[rank]
        pv = M[rank][col]
        for r in range(R):
            if r != rank and M[r][col] != 0:
                f = M[r][col] / pv
                for cc in range(C):
                    M[r][cc] -= f * M[rank][cc]
        rank += 1
        if rank == R:
            break
    return rank

def main():
    t = int(sys.argv[1])
    if t < 1: t = 1
    if t > 10: t = 10
    rng = random.Random(90437 * t + 13)

    a = 4 + t
    b = 4 + t
    c = 3 + t

    def rvec(n):
        return [rng.randint(-3, 3) for _ in range(n)]

    # T[i][j][k]
    T = [[[0] * c for _ in range(b)] for _ in range(a)]
    for k in range(c):
        # draw a rank-2 slice with all nonzero fibres relative to previous ones
        while True:
            p = rvec(a); q = rvec(b); s = rvec(a); tt = rvec(b)
            M = [[p[i] * q[j] + s[i] * tt[j] for j in range(b)] for i in range(a)]
            if rank2_int(M) != 2:
                continue
            break
        for i in range(a):
            for j in range(b):
                T[i][j][k] = M[i][j]

    # guarantee every (i,j) mode-c fibre is nonzero so the trivial baseline is
    # exactly a*b (deterministic given the seed above; if not, perturb slice 0)
    for i in range(a):
        for j in range(b):
            if all(T[i][j][k] == 0 for k in range(c)):
                T[i][j][0] += 1  # deterministic nudge (extremely rare)

    out = []
    out.append("%d %d %d" % (a, b, c))
    for k in range(c):
        for i in range(a):
            out.append(" ".join(str(T[i][j][k]) for j in range(b)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
