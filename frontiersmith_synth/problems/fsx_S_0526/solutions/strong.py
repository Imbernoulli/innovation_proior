# TIER: strong
# THE INSIGHT (affine-layer separation): a boolean permutation's cheap part is
# exactly its GF(2)-affine component.  Peel it straight off the table:
#     b = P(0),   column i of M = P(e_i) XOR b,
# because the planted nonlinear residue T fixes every weight<=1 vector. Then
#     T = A^{-1} o P,      T(x) = M^{-1} (P(x) XOR b),
# is a small triangular quadratic map. Realize the circuit as [T's few Toffolis]
# then [M as a cheap CNOT network] then [b as NOTs]. The Toffoli cost collapses
# onto the t-monomial residue -- no compute/uncompute, no per-bit duplication.
# Falls back to the structure-blind synthesis if the peel is not clean.
import sys


def bits_of(m):
    r = []
    i = 0
    while m:
        if m & 1:
            r.append(i)
        m >>= 1; i += 1
    return r


def anf_bits(F, n):
    N = 1 << n
    res = []
    for j in range(n):
        a = [(F[x] >> j) & 1 for x in range(N)]
        for i in range(n):
            bit = 1 << i
            for x in range(N):
                if x & bit:
                    a[x] ^= a[x ^ bit]
        res.append([m for m in range(N) if a[m]])
    return res


def invert_gf2(rows, n):
    # rows: n ints, bit c of rows[r] = M[r][c]. Return inverse rows or None.
    A = [rows[i] | (1 << (n + i)) for i in range(n)]  # augment with identity
    for col in range(n):
        piv = -1
        for r in range(col, n):
            if (A[r] >> col) & 1:
                piv = r; break
        if piv < 0:
            return None
        A[col], A[piv] = A[piv], A[col]
        for r in range(n):
            if r != col and ((A[r] >> col) & 1):
                A[r] ^= A[col]
    inv = [(A[r] >> n) & ((1 << n) - 1) for r in range(n)]
    return inv


def synth_linear_cnots(rows, n):
    # realize reg := M*reg (M given by rows) with CNOTs. Returns list of (ctrl,tgt).
    A = rows[:]
    ops = []  # (t,c): row t += row c
    for col in range(n):
        if not ((A[col] >> col) & 1):
            piv = next(r for r in range(col + 1, n) if (A[r] >> col) & 1)
            A[col] ^= A[piv]; ops.append((col, piv))
            A[piv] ^= A[col]; ops.append((piv, col))
            A[col] ^= A[piv]; ops.append((col, piv))
        for r in range(n):
            if r != col and ((A[r] >> col) & 1):
                A[r] ^= A[col]; ops.append((r, col))
    return [(c, t) for (t, c) in reversed(ops)]


def matvec(rows, v, n):
    z = 0
    for j in range(n):
        if bin(rows[j] & v).count("1") & 1:
            z |= (1 << j)
    return z


def fallback(P, n):
    Pinv = [0] * (1 << n)
    for x in range(1 << n):
        Pinv[P[x]] = x
    out = []
    for anf, cb, tb in ((anf_bits(P, n), 0, n),):
        for j in range(n):
            for m in anf[j]:
                bs = bits_of(m)
                if not bs:
                    out.append("NOT %d" % (tb + j))
                elif len(bs) == 1:
                    out.append("CNOT %d %d" % (cb + bs[0], tb + j))
                else:
                    out.append("TOF %d %d %d" % (cb + bs[0], cb + bs[1], tb + j))
    for i in range(n):
        out += ["CNOT %d %d" % (i, n + i), "CNOT %d %d" % (n + i, i), "CNOT %d %d" % (i, n + i)]
    anfPi = anf_bits(Pinv, n)
    for j in range(n):
        for m in anfPi[j]:
            bs = bits_of(m)
            if not bs:
                out.append("NOT %d" % (n + j))
            elif len(bs) == 1:
                out.append("CNOT %d %d" % (bs[0], n + j))
            else:
                out.append("TOF %d %d %d" % (bs[0], bs[1], n + j))
    return out


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); a = int(next(it))
    int(next(it)); int(next(it)); int(next(it))
    N = 1 << n
    P = [int(next(it)) for _ in range(N)]

    b = P[0]
    # column i of M
    cols = [P[1 << i] ^ b for i in range(n)]
    # rows[j] : bit i = (cols[i]>>j)&1
    rows = [0] * n
    for i in range(n):
        for j in range(n):
            if (cols[i] >> j) & 1:
                rows[j] |= (1 << i)

    Minv = invert_gf2(rows, n)
    if Minv is None:
        sys.stdout.write("\n".join(fallback(P, n)) + "\n"); return

    # T(x) = Minv * (P(x) ^ b)
    T = [matvec(Minv, P[x] ^ b, n) for x in range(N)]
    anfT = anf_bits(T, n)

    # extract Toffolis: bit k must be x_k XOR (sum of quadratic monomials).
    toffs = []
    clean = True
    for k in range(n):
        for m in anfT[k]:
            bs = bits_of(m)
            if len(bs) == 1 and bs[0] == k:
                continue  # identity linear part
            elif len(bs) == 2:
                toffs.append((bs[0], bs[1], k))
            else:
                clean = False
    if not clean:
        sys.stdout.write("\n".join(fallback(P, n)) + "\n"); return

    out = []
    # 1) apply T (the residue Toffolis) on the data register (reg = x)
    for (c0, c1, tg) in toffs:
        out.append("TOF %d %d %d" % (c0, c1, tg))
    # 2) apply M as a CNOT network: reg := M*reg
    for (c, tg) in synth_linear_cnots(rows, n):
        out.append("CNOT %d %d" % (c, tg))
    # 3) apply the affine offset b as NOTs
    for j in range(n):
        if (b >> j) & 1:
            out.append("NOT %d" % j)

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
