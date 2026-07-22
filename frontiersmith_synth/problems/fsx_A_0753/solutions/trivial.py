# TIER: trivial
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


BASELINE_EXTRA_MODULUS = 6


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

    # Reproduce the checker's own reference construction exactly (calibrates to ~0.1).
    nbits = bits_of(K)
    m_strong = max(nbits, Bmax + 1)
    m_base = m_strong + BASELINE_EXTRA_MODULUS
    cws = comb_codebook(L, K, m_base)

    out = []
    for v in cws:
        out.append(format(v, "0%db" % L))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
