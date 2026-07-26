# TIER: strong
# The insight (not "greedy + tuning"): re-index through TWO number-theoretic
# bijections instead of micro-optimizing the recursion.
#
#  1) Good-Thomas / Prime-Factor-Algorithm CRT mapping: because our stage lengths
#     are PAIRWISE COPRIME, indices can be re-labelled via the Chinese Remainder
#     Theorem (a Ruritanian map on the input side, a true CRT map on the output
#     side) so the n-point transform becomes an EXACTLY SEPARABLE multi-dimensional
#     DFT with NO twiddle factor between stages at all -- twiddle-elimination falls
#     out of the re-indexing for free, it is not hunted for case by case.
#  2) Rader's trick on every PRIME-length leaf: reindex both input and output by
#     powers of a primitive root mod p, turning the (p-1)x(p-1) "AC" block of the
#     prime DFT into a length-(p-1) cyclic convolution with a FIXED kernel (still
#     linear in the data). That convolution is then itself re-attacked with the
#     SAME CRT-splitting idea: since z^m - 1 = (z^(m/2)-1)(z^(m/2)+1) for even m,
#     the ring Z[z]/(z^m-1) splits (CRT again) into a half-size CYCLIC convolution
#     and a half-size NEGACYCLIC convolution, each done directly. The /2 rescaling
#     needed to invert that CRT split is folded into the (precomputed, free) kernel
#     constants rather than paid at runtime, so the split is genuinely free and
#     roughly HALVES the leaf's multiplication count versus a naive p x p DFT.
import sys
from math import gcd


class Builder:
    def __init__(self, n, q):
        self.n = n
        self.q = q
        self.instrs = []
        self.next_reg = n

    def add(self, a, b):
        self.instrs.append("A %d %d" % (a, b))
        r = self.next_reg
        self.next_reg += 1
        return r

    def sub(self, a, b):
        self.instrs.append("S %d %d" % (a, b))
        r = self.next_reg
        self.next_reg += 1
        return r

    def mulc(self, a, c):
        c %= self.q
        self.instrs.append("M %d %d" % (a, c))
        r = self.next_reg
        self.next_reg += 1
        return r

    def term(self, c, a):
        c %= self.q
        if c == 1:
            return a
        return self.mulc(a, c)

    def matvec(self, M, xregs):
        out = []
        for row in M:
            terms = [(c % self.q, xregs[j]) for j, c in enumerate(row) if c % self.q != 0]
            if not terms:
                out.append(self.mulc(xregs[0], 0))
                continue
            c0, r0 = terms[0]
            cur = self.term(c0, r0)
            for c, r in terms[1:]:
                t = self.term(c, r)
                cur = self.add(cur, t)
            out.append(cur)
        return out


def smallest_prime_factor(x):
    d = 2
    while d * d <= x:
        if x % d == 0:
            return d
        d += 1
    return x


def find_primitive_root(p):
    if p == 2:
        return 1
    order = p - 1
    fs = set()
    t = order
    d = 2
    while d * d <= t:
        while t % d == 0:
            fs.add(d)
            t //= d
        d += 1
    if t > 1:
        fs.add(t)
    g = 2
    while True:
        if all(pow(g, order // f, p) != 1 for f in fs):
            return g
        g += 1


def cyclic_conv_split(b, q, xdata, kernel, m):
    """Registers holding y[t] = sum_s xdata[s]*kernel[(t-s) mod m], t=0..m-1."""
    if m == 1:
        return [b.term(kernel[0] % q, xdata[0])]
    if m % 2 == 1:
        M = [[kernel[(t - s) % m] for s in range(m)] for t in range(m)]
        return b.matvec(M, xdata)

    r = m // 2
    inv2 = pow(2, -1, q)

    X0 = [b.add(xdata[i], xdata[i + r]) for i in range(r)]
    X1 = [b.sub(xdata[i], xdata[i + r]) for i in range(r)]

    H0 = [((kernel[i] + kernel[i + r]) * inv2) % q for i in range(r)]
    H1 = [((kernel[i] - kernel[i + r]) * inv2) % q for i in range(r)]

    M0 = [[H0[(t - s) % r] for s in range(r)] for t in range(r)]
    Y0 = b.matvec(M0, X0)

    M1 = [[(H1[(t - s) % r] if t >= s else (q - H1[(t - s) % r]) % q) for s in range(r)] for t in range(r)]
    Y1 = b.matvec(M1, X1)

    out = [None] * m
    for i in range(r):
        out[i] = b.add(Y0[i], Y1[i])
        out[i + r] = b.sub(Y0[i], Y1[i])
    return out


def rader_prime_dft(b, p, q, w, xregs):
    if p == 2:
        return [b.add(xregs[0], xregs[1]), b.sub(xregs[0], xregs[1])]

    g = find_primitive_root(p)
    m = p - 1
    gt = [pow(g, t, p) for t in range(m)]
    # reversed index (x'[s] = x[g^{-s}]) so that the required "correlation" sum
    # sum_s x[g^s]*c[(s+t) mod m] becomes a STANDARD cyclic convolution sum_s
    # x'[s]*c[(t-s) mod m], matching cyclic_conv_split's convention.
    xprime = [xregs[gt[(-s) % m]] for s in range(m)]
    c = [pow(w, gt[u], q) for u in range(m)]

    X0 = xregs[0]
    for t in range(1, p):
        X0 = b.add(X0, xregs[t])

    yregs = cyclic_conv_split(b, q, xprime, c, m)

    out = [None] * p
    out[0] = X0
    for t in range(m):
        k = gt[t]
        out[k] = b.add(xregs[0], yregs[t])
    return out


def pfa_transform(b, n, q, w, xregs):
    if n == 1:
        return [xregs[0]]
    N1 = smallest_prime_factor(n)
    if N1 == n:
        return rader_prime_dft(b, n, q, w, xregs)

    N2 = n // N1
    assert gcd(N1, N2) == 1

    def in_idx(n1, n2):
        return (N2 * n1 + N1 * n2) % n

    wp1 = pow(w, N2, q)  # primitive N1-th root
    wp2 = pow(w, N1, q)  # primitive N2-th root

    stageA = []
    for n1 in range(N1):
        sub_x = [xregs[in_idx(n1, n2)] for n2 in range(N2)]
        stageA.append(pfa_transform(b, N2, q, wp2, sub_x))

    Y = [[None] * N2 for _ in range(N1)]
    for k2 in range(N2):
        col = [stageA[n1][k2] for n1 in range(N1)]
        col_out = pfa_transform(b, N1, q, wp1, col)
        for k1 in range(N1):
            Y[k1][k2] = col_out[k1]

    invN2modN1 = pow(N2, -1, N1) if N1 > 1 else 0
    invN1modN2 = pow(N1, -1, N2) if N2 > 1 else 0
    out = [None] * n
    for k1 in range(N1):
        for k2 in range(N2):
            k = (k1 * N2 * invN2modN1 + k2 * N1 * invN1modN2) % n
            out[k] = Y[k1][k2]
    return out


def main():
    n, q, w = map(int, sys.stdin.read().split())
    xregs = list(range(n))
    b = Builder(n, q)
    outs = pfa_transform(b, n, q, w, xregs)

    L = len(b.instrs)
    out = ["%d %d" % (n + L, L)]
    out.extend(b.instrs)
    out.append("O " + " ".join(map(str, outs)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
