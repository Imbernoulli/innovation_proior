# TIER: strong
import sys

# The insight: pi = L o sigma o L^{-1} for some invertible GF(2)-linear L and some sigma
# that is local to a small subset of coordinates. This solution never guesses at L (that
# would be a huge search); instead it DERIVES it directly from pi's own behaviour.
#
# Key fact: if h lies in W := L(span of sigma's untouched coordinates), then for EVERY x,
#   pi(x xor h) == pi(x) xor h.
# (Adding h to x only perturbs the untouched part of x's L^{-1}-image, which sigma passes
# straight through, so the perturbation survives unchanged all the way to the output.)
# W is provably a GF(2)-linear subspace. So: sweep every candidate h, keep the ones that
# satisfy the identity for ALL x, and take their linear span -- this recovers W exactly,
# with NO prior knowledge of L, sigma, or the touched-coordinate count k. This is a global
# search over a whole vector space of candidate "conjugating frames", not a search over
# gate sequences -- exactly the reframing the family rewards.
#
# Once W (dimension d = n-k) is known: extend a basis of W to a full basis of F_2^n with
# k extra directions; that change of basis P turns pi into "leave the first d coordinates
# alone, permute the last k coordinates via some s (plus, in general, an XOR offset into
# the first d coordinates that also only depends on those last k bits)". P and P^{-1} are
# realized cheaply as CNOT-only circuits via Gaussian elimination (O(n^2)); s (and any
# offset) lives on only 2^k points and is realized with the same cheap transposition
# gadget used elsewhere, restricted to k register bits instead of n.


def gate_weight(k):
    if k == 0:
        return 1
    if k == 1:
        return 2
    return 5


class XorBasis:
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
        return False

    def dim(self):
        return sum(1 for x in self.basis if x)

    def vectors(self):
        return [x for x in self.basis if x]


def linear_map_to_cnots(cols, n):
    """cols[j] = image of e_j under the desired map M; returns a CNOT gate list (single
    positive control each) realizing register x -> M(x), via swap-free Gauss-Jordan
    elimination of M's row representation down to the identity (row ops are self-inverse,
    so applying the SAME ops to the register in reverse order goes identity -> M)."""
    rows = [0] * n
    for j in range(n):
        for i in range(n):
            if (cols[j] >> i) & 1:
                rows[i] |= (1 << j)
    work = rows[:]
    ops = []
    for c in range(n):
        if not ((work[c] >> c) & 1):
            r = next(r for r in range(c + 1, n) if (work[r] >> c) & 1)
            work[c] ^= work[r]; ops.append((r, c))
        for r in range(c + 1, n):
            if (work[r] >> c) & 1:
                work[r] ^= work[c]; ops.append((c, r))
    for c in range(n - 1, -1, -1):
        for r in range(c):
            if (work[r] >> c) & 1:
                work[r] ^= work[c]; ops.append((c, r))
    return [([(a, 1)], b) for (a, b) in reversed(ops)]


def transposition_gates_on(bits, v1, v2):
    """v1,v2 are LOCAL values over the register bits listed in `bits` (bits[0]=local bit0,
    ...); realizes an exact transposition of these two local values, all others fixed."""
    m = len(bits)
    diff = [i for i in range(m) if ((v1 >> i) & 1) != ((v2 >> i) & 1)]
    fwd = []
    cur = v1
    for i in diff:
        w_prev = cur
        controls = [(bits[b], (w_prev >> b) & 1) for b in range(m) if b != i]
        fwd.append((controls, bits[i]))
        cur ^= (1 << i)
    return fwd + list(reversed(fwd[:-1])) if fwd else []


def local_perm_gates(bits, perm):
    m = len(bits)
    Nk = 1 << m
    cur = list(range(Nk))
    pos = list(range(Nk))
    gates = []
    for x in range(Nk):
        target = perm[x]
        if cur[x] == target:
            continue
        y = pos[target]
        gates.extend(transposition_gates_on(bits, cur[x], cur[y]))
        vx, vy = cur[x], cur[y]
        cur[x], cur[y] = cur[y], cur[x]
        pos[vx], pos[vy] = y, x
    return gates


def solve(pi, n):
    N = 1 << n
    wb = XorBasis(n)
    for h in range(1, N):
        if all(pi[x ^ h] == (pi[x] ^ h) for x in range(N)):
            wb.insert(h)
    basisW = wb.vectors()
    d = len(basisW)

    full = XorBasis(n)
    for v in basisW:
        full.insert(v)
    ext = []
    for e in range(n):
        if full.insert(1 << e):
            ext.append(1 << e)

    Pcols = basisW + ext  # n columns total: first d = W-basis, last k_eff = extension
    k_eff = n - d

    def apply_P(y):
        r = 0
        for i in range(n):
            if (y >> i) & 1:
                r ^= Pcols[i]
        return r

    table = {}
    for y in range(N):
        table[apply_P(y)] = y

    def apply_Pinv(x):
        return table[x]

    encode_cols = [apply_Pinv(1 << j) for j in range(n)]  # M(e_j) for M = P^{-1}
    encode_gates = linear_map_to_cnots(encode_cols, n)
    decode_gates = linear_map_to_cnots(Pcols, n)  # M(e_j)=Pcols[j] for M = P

    Nk = 1 << k_eff
    s = [0] * Nk
    offset_gates = []
    local_bits = list(range(d, n))
    for y2 in range(Nk):
        y_full = y2 << d
        x = apply_P(y_full)
        outy = apply_Pinv(pi[x])
        c_part = outy & ((1 << d) - 1)
        s[y2] = outy >> d
        if c_part:
            for i in range(d):
                if (c_part >> i) & 1:
                    controls = [(d + b, (y2 >> b) & 1) for b in range(k_eff)]
                    offset_gates.append((controls, i))

    inner_gates = local_perm_gates(local_bits, s)

    return encode_gates + offset_gates + inner_gates + decode_gates


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    N = 1 << n
    pi = [int(next(it)) for _ in range(N)]

    gates = solve(pi, n)

    out = [str(len(gates))]
    for controls, t in gates:
        parts = [str(len(controls))]
        for c, p in controls:
            parts.append(str(c)); parts.append(str(p))
        parts.append(str(t))
        out.append(" ".join(parts))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
