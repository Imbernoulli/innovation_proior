import sys, random

# gen.py <testId> -- prints ONE "conjugated permutation" instance to stdout.
#
# Plant: pi = L o sigma o L^{-1} where L is an invertible linear (GF(2)) map on n-bit
# strings and sigma is a permutation that depends on only k << n of the (post-L^{-1})
# coordinates and passes the remaining n-k coordinates through unchanged. In the raw
# x-basis pi looks like an arbitrary, unstructured permutation on N=2^n states; only in
# the L-conjugated basis does it collapse to "leave n-k bits alone, shuffle k bits".
#
# Rejection sampling guarantees (both checked, both re-rolled from the SAME seeded rng
# so generation stays fully deterministic):
#   (a) L is genuinely invertible over GF(2).
#   (b) The invariant subspace W = L(span of the untouched coordinates) contains NO
#       single standard basis vector -- i.e. the "obvious" per-axis check (does flipping
#       bit e alone commute with pi?) NEVER succeeds. This is what makes the naive/greedy
#       approaches blind to the structure: only a genuine change-of-basis search finds it.
#   (c) sigma's k-bit inner permutation has no accidental XOR/affine symmetry of its own
#       (which for small k could otherwise leak extra "easy" directions into W).

LADDER = {
    1: (5, 2), 2: (5, 3), 3: (6, 2), 4: (6, 3), 5: (6, 4),
    6: (7, 2), 7: (7, 3), 8: (7, 4), 9: (8, 3), 10: (9, 4),
}
BASE_SEED = 987654321


class XorBasis:
    """Standard GF(2) linear (xor) basis: basis[b] is either 0 or a vector whose
    highest set bit is exactly b."""
    def __init__(self, n):
        self.basis = [0] * n
        self.n = n

    def insert(self, v):
        for b in range(self.n - 1, -1, -1):
            if not ((v >> b) & 1):
                continue
            if self.basis[b] == 0:
                self.basis[b] = v
                return True
            v ^= self.basis[b]
        return False  # v was already in the span

    def reduce(self, v):
        for b in range(self.n - 1, -1, -1):
            if (v >> b) & 1 and self.basis[b]:
                v ^= self.basis[b]
        return v

    def dim(self):
        return sum(1 for x in self.basis if x)

    def vectors(self):
        return [x for x in self.basis if x]


def gen_conjugated_perm(n, k, seed):
    rng = random.Random(seed)
    N = 1 << n
    while True:
        L = [rng.getrandbits(n) for _ in range(n)]
        xb = XorBasis(n)
        if not all(xb.insert(v) for v in L) or xb.dim() != n:
            continue
        wb = XorBasis(n)
        for v in L[k:]:
            wb.insert(v)
        if any(wb.reduce(1 << e) == 0 for e in range(n)):
            continue
        break

    S = list(range(k))
    Nk = 1 << k
    while True:
        pv = list(range(Nk))
        rng.shuffle(pv)
        bad = False
        for d in range(1, Nk):
            if all(pv[y ^ d] == (pv[y] ^ d) for y in range(Nk)):
                bad = True
                break
        if not bad:
            break
    sigmaS = pv

    def apply_L(x):
        r = 0
        for i in range(n):
            if (x >> i) & 1:
                r ^= L[i]
        return r

    table = {}
    for x in range(N):
        table[apply_L(x)] = x
    Linv = [table[1 << j] for j in range(n)]

    def apply_Linv(y):
        r = 0
        for i in range(n):
            if (y >> i) & 1:
                r ^= Linv[i]
        return r

    def sigma(y):
        yS = 0
        for idx, bit in enumerate(S):
            if (y >> bit) & 1:
                yS |= (1 << idx)
        yS2 = sigmaS[yS]
        out = y
        for idx, bit in enumerate(S):
            if (yS2 >> idx) & 1:
                out |= (1 << bit)
            else:
                out &= ~(1 << bit)
        return out

    pi = [apply_L(sigma(apply_Linv(x))) for x in range(N)]
    return pi


def main():
    tid = int(sys.argv[1])
    n, k = LADDER[tid]
    pi = gen_conjugated_perm(n, k, seed=BASE_SEED + tid * 1000003)
    out = [str(n), " ".join(str(v) for v in pi)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
