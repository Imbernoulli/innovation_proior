#!/usr/bin/env python3
"""gen.py <testId>  ->  prints ONE deep-space-downlink decoding instance to stdout.

An instance is a fixed binary LDPC (low-density parity-check) code, given by its
parity-check matrix H (r x n over GF(2)), plus a batch of m received frames, each an
n-bit word that arrived over a noisy channel (a codeword XORed with a bit-flip error
pattern).  The solver must decode every frame back to a codeword.

testId 1..10 is a fixed difficulty ladder (small -> large block length / batch). All
randomness is seeded ONLY by testId, so the instance is bit-for-bit deterministic. The
ground truth (which codeword each frame really came from) is NOT emitted -- only H and
the received words -- so the solver must actually decode.
"""
import sys

# ------------------------------------------------------------------ deterministic RNG
class XorShift64:
    def __init__(self, seed):
        self.s = (seed ^ 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        if self.s == 0:
            self.s = 0x1234567890ABCDEF

    def next(self):
        x = self.s
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 7)
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        self.s = x & 0xFFFFFFFFFFFFFFFF
        return self.s

    def randint(self, a, b):          # inclusive
        return a + self.next() % (b - a + 1)

    def sample(self, pop, k):         # k distinct ints from range(pop)
        chosen = set()
        while len(chosen) < k:
            chosen.add(self.next() % pop)
        return list(chosen)

    def shuffle(self, arr):
        for i in range(len(arr) - 1, 0, -1):
            j = self.next() % (i + 1)
            arr[i], arr[j] = arr[j], arr[i]


# ------------------------------------------------------------------ GF(2) linear algebra
def rref_gf2(rows, n):
    """Reduced row echelon form over GF(2). Returns (reduced_rows, pivot_cols)."""
    rows = list(rows)
    m = len(rows)
    pivot_cols = []
    r = 0
    for col in range(n):
        piv = -1
        for i in range(r, m):
            if (rows[i] >> col) & 1:
                piv = i
                break
        if piv == -1:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        pr = rows[r]
        for i in range(m):
            if i != r and ((rows[i] >> col) & 1):
                rows[i] ^= pr
        pivot_cols.append(col)
        r += 1
        if r == m:
            break
    return rows[:r], pivot_cols


def nullspace_basis(rows, n):
    """A basis (list of n-bit ints) for { x in GF(2)^n : H x = 0 }."""
    red, pivot_cols = rref_gf2(rows, n)
    pivset = set(pivot_cols)
    free = [c for c in range(n) if c not in pivset]
    basis = []
    for f in free:
        v = 1 << f
        for ri, pc in enumerate(pivot_cols):
            if (red[ri] >> f) & 1:
                v |= (1 << pc)
        basis.append(v)
    return basis


def parity(x):
    return bin(x).count("1") & 1


def to_bits(v, n):
    return "".join("1" if (v >> j) & 1 else "0" for j in range(n))


# ------------------------------------------------------------------ instance ladder
#   (n = block length [even], m = batch size).  Column weight is fixed at 3, rate ~1/2
#   (r = n/2 parity checks), decoding radius T = 8.
LADDER = [
    (72,  200),
    (96,  220),
    (120, 240),
    (150, 260),
    (180, 280),
    (210, 300),
    (240, 320),
    (270, 340),
    (288, 360),
    (300, 400),
]

COL_WEIGHT = 3
T_RADIUS = 8

# error-weight mix (fractions of the batch).  Weights 0..6 are inside the decoding
# radius (correctable in principle); the remainder are heavy bursts far beyond it.
MIX = [
    (0, 0.10),
    (1, 0.18),
    (2, 0.14),
    (3, 0.12),
    (4, 0.10),
    (5, 0.08),
    (6, 0.06),
]
HEAVY_RANGE = (12, 18)   # uncorrectable headroom bursts


def build_H(rng, n, r):
    """Column-regular LDPC: every code coordinate participates in exactly COL_WEIGHT
    parity checks; check rows have irregular but bounded weight."""
    H = [0] * r
    cols = []
    for j in range(n):
        chk = rng.sample(r, COL_WEIGHT)
        cols.append(chk)
        for t in chk:
            H[t] |= (1 << j)
    return H


def build_weights(rng, m):
    weights = []
    for w, frac in MIX:
        weights.extend([w] * round(frac * m))
    while len(weights) < m:
        weights.append(rng.randint(HEAVY_RANGE[0], HEAVY_RANGE[1]))
    weights = weights[:m]
    # guarantee at least one clean (weight-0) frame so the trivial baseline is positive
    if 0 not in weights:
        weights[0] = 0
    rng.shuffle(weights)
    return weights


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    t = int(sys.argv[1])
    if t < 1:
        t = 1
    if t > len(LADDER):
        n, m = LADDER[-1]
    else:
        n, m = LADDER[t - 1]
    r = n // 2

    rng = XorShift64(0xC0FFEE * (t + 1) + 12345)

    H = build_H(rng, n, r)
    basis = nullspace_basis(H, n)
    dim = len(basis)

    weights = build_weights(rng, m)

    out = []
    out.append("%d %d %d %d" % (n, r, m, T_RADIUS))
    for row in H:
        out.append(to_bits(row, n))

    for w in weights:
        # random codeword = random combination of null-space basis vectors
        c = 0
        for b in basis:
            if rng.next() & 1:
                c ^= b
        # error pattern of exactly weight w, and never itself a codeword
        if w == 0:
            e = 0
        else:
            while True:
                pos = rng.sample(n, w)
                e = 0
                for p in pos:
                    e |= (1 << p)
                # reject an error that is itself a codeword (would masquerade as clean)
                bad = any(parity(H[tt] & e) for tt in range(r))
                if bad:  # bad==True means syndrome != 0  -> NOT a codeword  -> keep
                    break
        y = c ^ e
        out.append(to_bits(y, n))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
