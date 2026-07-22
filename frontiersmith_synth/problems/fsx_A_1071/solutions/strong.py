# TIER: strong
"""The insight: put a SINGLE length-s sequence on a circulant, and the whole
n x n orthogonality problem collapses to that one sequence's periodic
autocorrelation. If c = (c_0,...,c_{s-1}) is the first row and every row is a
cyclic shift of c, then (W W^T)_{i,j} depends only on (i-j) mod s -- it is the
autocorrelation A(d) = sum_t c[t] c[t+d mod s]. So W W^T = k*I EXACTLY iff
A(d) = 0 for every shift d != 0.

Quadratic-residue sign patterns kill that autocorrelation by character-sum
cancellation: for a prime p = 3 (mod 4), the Legendre-symbol circulant
(Jacobsthal matrix) Q has off-peak autocorrelation constant at -1, and
bordering it with an all-ones row/column turns that constant defect into an
EXACT cancellation -- the classical Paley conference-matrix identity
C C^T = p * I for the (p+1) x (p+1) matrix C. So whenever k = p is such a
prime, a (p+1) x (p+1) perfect weighing block exists with zero defect.

For n a multiple of s = k+1 this block can be TILED (block-diagonal) -- the
off-block entries are all 0 so cross terms between different blocks vanish
automatically, patching cases where a single circulant of length n has no
such structure. When n is not an exact multiple of s, tile as many perfect
blocks as fit and cover the leftover rows with a plain sliding window (a
local, not globally optimal, patch -- but still far better than ignoring the
block structure entirely)."""
import sys


def legendre(a, p):
    a %= p
    if a == 0:
        return 0
    r = pow(a, (p - 1) // 2, p)
    return 1 if r == 1 else -1


def conf_block(p):
    """(p+1) x (p+1) matrix with entries in {-1,0,1}, row weight p,
    C C^T = p * I exactly (Paley conference-matrix construction), valid for
    prime p with p % 4 == 3."""
    s = p + 1
    C = [[0] * s for _ in range(s)]
    for j in range(1, s):
        C[0][j] = 1
        C[j][0] = 1
    for i in range(1, s):
        for j in range(1, s):
            C[i][j] = legendre(j - i, p)
    return C


def is_prime(x):
    if x < 2:
        return False
    if x % 2 == 0:
        return x == 2
    i = 3
    while i * i <= x:
        if x % i == 0:
            return False
        i += 2
    return True


def main():
    n, k = map(int, sys.stdin.read().split()[:2])
    W = [[0] * n for _ in range(n)]

    p = k
    if p >= 3 and p % 4 == 3 and is_prime(p):
        s = p + 1
        block = conf_block(p)
        m = n // s
        r = n - m * s
        for b in range(m):
            off = b * s
            for i in range(s):
                for j in range(s):
                    v = block[i][j]
                    if v:
                        W[off + i][off + j] = v
        if r > 0:
            # leftover rows: sliding window fallback (still exactly k nonzero,
            # still feasible -- just not provably orthogonal)
            for i in range(m * s, n):
                for t in range(k):
                    W[i][(i + t) % n] = 1
    else:
        # k is not a usable prime for this construction on this input --
        # fall back to the sliding-window baseline (still feasible).
        for i in range(n):
            for t in range(k):
                W[i][(i + t) % n] = 1

    out = "\n".join(" ".join(map(str, row)) for row in W)
    sys.stdout.write(out + "\n")


if __name__ == "__main__":
    main()
