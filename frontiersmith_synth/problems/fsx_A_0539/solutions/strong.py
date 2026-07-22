# TIER: strong
# The insight: the care set is a low-dimensional affine flat, and the target,
# read in free coordinates, is a quadratic form of LOW symplectic rank.  Detect
# the flat, take the Reed-Muller form, then reduce the alternating quadratic form
# to its symplectic normal form  q = XOR_p f_p * h_p  with only rank/2 products.
# The number of AND gates collapses from ~(monomials) to rank/2, far below what
# per-monomial or shared-variable factoring achieves.
import sys


def analyze():
    toks = sys.stdin.read().split()
    it = iter(toks)
    n = int(next(it)); m = int(next(it))
    xs = []; ys = []
    for _ in range(m):
        bits = next(it); y = int(next(it))
        X = 0
        for i, ch in enumerate(bits):
            if ch == '1':
                X |= (1 << i)
        xs.append(X); ys.append(y)
    X0 = xs[0]
    deltas = [x ^ X0 for x in xs]
    basis = []
    for d in deltas:
        v = d
        for b in basis:
            lb = b.bit_length() - 1
            if (v >> lb) & 1:
                v ^= b
        if v:
            basis.append(v)
    B = basis[:]
    pivbits = []; r = 0
    for col in range(n - 1, -1, -1):
        sel = -1
        for row in range(r, len(B)):
            if (B[row] >> col) & 1:
                sel = row; break
        if sel < 0:
            continue
        B[r], B[sel] = B[sel], B[r]
        for row in range(len(B)):
            if row != r and (B[row] >> col) & 1:
                B[row] ^= B[r]
        pivbits.append(col); r += 1
        if r == len(B):
            break
    k = len(B)
    ytab = [0] * (1 << k)
    for p in range(m):
        xv = xs[p]; t = 0
        for i, pb in enumerate(pivbits):
            if (xv >> pb) & 1:
                t |= (1 << i)
        ytab[t] = ys[p]
    anf = ytab[:]
    for i in range(k):
        bit = 1 << i
        for mask in range(1 << k):
            if mask & bit:
                anf[mask] ^= anf[mask ^ bit]
    return n, m, k, pivbits, anf


class Circuit:
    def __init__(self, n):
        self.n = n
        self.lines = []

    def emit(self, s):
        self.lines.append(s)
        return self.n + len(self.lines) - 1

    def AND(self, a, b):
        return self.emit("AND %d %d" % (a, b))

    def XOR(self, a, b):
        return self.emit("XOR %d %d" % (a, b))

    def ONE(self):
        return self.emit("ONE")

    def ZERO(self):
        return self.emit("ZERO")

    def dump(self, out):
        sys.stdout.write("%d\n" % len(self.lines))
        if self.lines:
            sys.stdout.write("\n".join(self.lines) + "\n")
        sys.stdout.write("OUT %d\n" % out)


def and_chain(c, ws):
    w = ws[0]
    for v in ws[1:]:
        w = c.AND(w, v)
    return w


def xor_tree(c, ws):
    w = ws[0]
    for v in ws[1:]:
        w = c.XOR(w, v)
    return w


def main():
    n, m, k, pivbits, anf = analyze()
    c = Circuit(n)
    const = anf[0]
    linmask = 0
    A = [0] * k
    extra_terms = []  # degree>=3 monomials, per-monomial (defensive; unused for shipped data)
    for mask in range(1, 1 << k):
        if not anf[mask]:
            continue
        pc = bin(mask).count("1")
        if pc == 1:
            linmask |= mask
        elif pc == 2:
            i = (mask & -mask).bit_length() - 1
            j = (mask ^ (1 << i)).bit_length() - 1
            A[i] |= (1 << j); A[j] |= (1 << i)
        else:
            vs = [pivbits[b] for b in range(k) if (mask >> b) & 1]
            extra_terms.append(and_chain(c, vs))

    # symplectic reduction: peel rank-2 pieces
    lin = linmask
    prods = []
    Aw = A[:]
    while any(Aw):
        i = next(ii for ii in range(k) if Aw[ii])
        j = (Aw[i] & -Aw[i]).bit_length() - 1
        cj = 0; ci = 0
        for l in range(k):
            if (Aw[l] >> j) & 1:
                cj |= (1 << l)
            if (Aw[l] >> i) & 1:
                ci |= (1 << l)
        prods.append((cj, ci))
        lin ^= (cj & ci)           # diagonal t_l^2 = t_l self-terms
        for l in range(k):
            add = 0
            if (cj >> l) & 1:
                add ^= ci
            if (ci >> l) & 1:
                add ^= cj
            Aw[l] ^= add

    terms = list(extra_terms)
    for (fm, hm) in prods:
        fw = xor_tree(c, [pivbits[b] for b in range(k) if (fm >> b) & 1])
        hw = xor_tree(c, [pivbits[b] for b in range(k) if (hm >> b) & 1])
        terms.append(c.AND(fw, hw))
    if lin:
        terms.append(xor_tree(c, [pivbits[b] for b in range(k) if (lin >> b) & 1]))
    if const:
        terms.append(c.ONE())
    if not terms:
        out = c.ZERO()
    else:
        out = xor_tree(c, terms)
    c.dump(out)


if __name__ == "__main__":
    main()
