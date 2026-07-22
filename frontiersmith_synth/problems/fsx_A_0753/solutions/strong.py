# TIER: strong
# Insight: the adversary's error set is not "any sparse vector" -- it is the
# published low-dimensional family of contiguous bursts (length Bmin..Bmax at
# every offset) plus composite double-bursts (b1,b2 separated by gap G at
# every offset). Instead of maximizing raw pairwise Hamming distance (which
# says nothing about the SHAPE of the pairwise differences), interleave each
# message-index-bit across an arithmetic-progression "comb" of positions with
# spacing m = max(ceil(log2 K), Bmax+1). Because the spacing exceeds every
# swept burst length, no single burst (and no half of a composite
# double-burst) can ever land on more than one position of a given comb, so a
# pairwise codeword difference can never be swallowed by a published pattern
# the way a single contiguous block can. This trades a little raw Hamming
# weight for immunity to the *specific* structured sweep the codebook will
# actually be judged against.
import sys


def bits_of(K):
    return max(1, (K - 1).bit_length())


def comb_codebook(L, K, modulus):
    nbits = bits_of(K)
    rows = []
    for r in range(modulus):
        v = 0
        p = r
        while p < L:
            v |= 1 << p
            p += modulus
        rows.append(v)
    cws = []
    for i in range(K):
        v = 0
        for r in range(nbits):
            if (i >> r) & 1:
                v ^= rows[r]
        cws.append(v)
    return cws


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    L = int(next(it))
    K = int(next(it))
    Bmin = int(next(it))
    Bmax = int(next(it))
    G = int(next(it))
    D = int(next(it))
    for _ in range(D):
        next(it)
        next(it)

    nbits = bits_of(K)
    m_strong = max(nbits, Bmax + 1)
    cws = comb_codebook(L, K, m_strong)

    out = [format(v, "0%db" % L) for v in cws]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
