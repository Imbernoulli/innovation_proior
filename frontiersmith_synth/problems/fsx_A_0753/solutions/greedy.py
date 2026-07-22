# TIER: greedy
# The "obvious" recipe: maximize raw pairwise Hamming distance by giving each
# message-index-bit its own dedicated CONTIGUOUS block of the codeword (a
# textbook block/one-hot-per-bit code). This looks great by classic
# minimum-distance intuition -- every pair of codewords differs in a whole
# block's worth of bits -- but it never looks at the SHAPE of the threat
# model. When the swept burst family is wide enough to fit inside (or
# straddle) one of these blocks, a single published burst can be fully
# absorbed by one block, collapsing the decode margin for that pair.
import sys


def bits_of(K):
    return max(1, (K - 1).bit_length())


def block_codebook(L, K):
    nbits = bits_of(K)
    base = L // nbits
    rem = L % nbits
    rows = []
    pos = 0
    for r in range(nbits):
        ln = base + (1 if r < rem else 0)
        v = 0
        for p in range(pos, pos + ln):
            v |= 1 << p
        rows.append(v)
        pos += ln
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
    for _ in range(4):
        next(it)  # Bmin Bmax G D (unused: this recipe never looks at the sweep)

    cws = block_codebook(L, K)
    out = [format(v, "0%db" % L) for v in cws]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
