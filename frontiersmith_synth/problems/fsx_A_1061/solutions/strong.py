# TIER: strong
# Lifts a Sidon set through the D | M subgroup chain instead of searching Z_M flat.
#
# Key fact this exploits (the innovation hook): write M = D * M0 with gcd(D, M0) = 1, so by
# CRT, Z_M = Z_D x Z_M0 as rings. If S is a genuine SIDON set of size k in Z_M0 (all pairwise
# sums distinct) and C is ANY length-k sequence in Z_D, and we build A by combining them
# INDEX-ALIGNED (A[i] = CRT(C[i], S[i])), then a+b == c+e (mod M) for indices (i,j,p,q) forces
# S[i]+S[j] == S[p]+S[q] (mod M0); Sidon-ness of S pins {p,q} = {i,j}. But those two matches
# ((p,q)=(i,j) or (j,i)) automatically satisfy the D-coordinate equation too (sums are
# commutative), so the D-component is *never* the bottleneck once S is Sidon: A is a true
# Sidon set in the FULL group Z_M (E_M(A) hits the exact floor 2k^2-k), REGARDLESS of what C
# is. That leaves C completely free to independently minimize the coarse term E_D(A) = E_D(C)
# -- so we set C to the provably-optimal as-equal-as-possible residue distribution over Z_D.
# One CRT lift controls both quotients at once; no flat single-modulus search does that.
import sys


def egcd(a, b):
    if b == 0:
        return a, 1, 0
    g, x1, y1 = egcd(b, a % b)
    return g, y1, x1 - (a // b) * y1


def modinv(a, n):
    g, x, _ = egcd(a % n, n)
    return x % n


def balanced_sequence(D, k):
    base, rem = divmod(k, D)
    seq = []
    for r in range(D):
        seq.extend([r] * (base + 1 if r < rem else base))
    return seq[:k]


def sidon_in_zn(n, k):
    """Deterministic left-to-right greedy Sidon set in Z_n (Mian-Chowla style)."""
    A = []
    sums = set()
    c = 0
    while len(A) < k and c < n:
        ok = True
        for a in A:
            if (a + c) % n in sums:
                ok = False
                break
        if ok:
            for a in A:
                sums.add((a + c) % n)
            sums.add((2 * c) % n)
            A.append(c)
        c += 1
    # Safety fallback (should not trigger given how gen.py sizes M0): pad with any unused
    # residues so we always emit exactly k values, even if a perfect Sidon set ran out.
    used = set(A)
    c = 0
    while len(A) < k and c < n:
        if c not in used:
            A.append(c)
            used.add(c)
        c += 1
    return A


def main():
    toks = sys.stdin.read().split()
    M, D, k, W = int(toks[0]), int(toks[1]), int(toks[2]), int(toks[3])
    M0 = M // D

    C = balanced_sequence(D, k)          # optimal coarse-quotient distribution, free choice
    S = sidon_in_zn(M0, k)               # true Sidon set in the large coprime factor

    invM0 = modinv(M0, D)
    invD = modinv(D, M0)
    A = []
    for i in range(k):
        x = (C[i] * M0 * invM0 + S[i] * D * invD) % M
        A.append(x)

    sys.stdout.write(" ".join(map(str, A)) + "\n")


if __name__ == "__main__":
    main()
