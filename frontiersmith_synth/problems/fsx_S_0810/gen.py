#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE training trace to stdout.

A hidden substitution (morphism) sigma over the alphabet {0,1,2} is applied
repeatedly to the axiom "0": w_0 = "0", w_{n+1} = concatenation of sigma(c)
for each letter c of w_n (in order). Each testId fixes a DIFFERENT hidden
sigma. STDOUT prints ONLY:

    t n_train
    w_0
    w_1
    ...
    w_{n_train}

sigma itself, the incidence matrix, its eigenvalue, and the random seed are
NEVER printed -- they live only inside gen.py/verify.py's shared, private
`make_morphism` routine, and are regenerated (never read from a file) by the
grader from testId alone.
"""
import sys, random

K = 3
ALPHABET = "012"
LMAX_TRUE = 4
BASE_SEED = 913461
MAX_TRIES = 500


def n_train(t):
    return 5 + (t - 1) % 4


def _bool_matmul(A, B):
    C = [[0] * K for _ in range(K)]
    for i in range(K):
        for k in range(K):
            if A[i][k]:
                for j in range(K):
                    if B[k][j]:
                        C[i][j] = 1
    return C


def _primitive(M):
    Bm = [[1 if M[i][j] > 0 else 0 for j in range(K)] for i in range(K)]
    P = [row[:] for row in Bm]
    for _ in range(K * K + 2):
        if all(all(x > 0 for x in row) for row in P):
            return True
        P = _bool_matmul(P, Bm)
    return all(all(x > 0 for x in row) for row in P)


def _spectral_radius(M, iters=400):
    v = [1.0 / K] * K
    for _ in range(iters):
        w = [sum(M[i][j] * v[j] for j in range(K)) for i in range(K)]
        s = sum(w)
        if s <= 0:
            return 0.0
        v = [x / s for x in w]
    w = [sum(M[i][j] * v[j] for j in range(K)) for i in range(K)]
    return sum(w)


def apply_morphism(images, s):
    return ''.join(images[int(c)] for c in s)


def build_levels(images, axiom, n):
    levels = [str(axiom)]
    for _ in range(n):
        levels.append(apply_morphism(images, levels[-1]))
    return levels


def make_morphism(t):
    """Deterministic hidden sigma for this test id. Retries (seeded purely by
    t + an attempt counter) until primitive, growing (1.2<=lambda<=3.6), and
    every letter has appeared by n_train-1 (so its image is identifiable from
    the printed training levels)."""
    nt = n_train(t)
    tries = 0
    while True:
        tries += 1
        if tries > MAX_TRIES:
            raise RuntimeError("no valid morphism for t=%d" % t)
        rng = random.Random(BASE_SEED + t * 7919 + tries * 104729)
        images = []
        for i in range(K):
            L = rng.randint(1, LMAX_TRUE)
            images.append(''.join(rng.choice(ALPHABET) for _ in range(L)))
        M = [[images[i].count(str(j)) for j in range(K)] for i in range(K)]
        if not _primitive(M):
            continue
        lam = _spectral_radius(M)
        if not (1.2 <= lam <= 3.6):
            continue
        levels = build_levels(images, 0, nt)
        seen = set(''.join(levels[:-1]))
        if len(seen) < K:
            continue
        return images, M, lam, nt, levels


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    images, M, lam, nt, levels = make_morphism(t)
    out = ["%d %d" % (t, nt)]
    out.extend(levels)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
