# TIER: greedy
# The natural first improvement: PEEL the GF(2)-affine shell (b=P(0), M columns
# from P(e_i), residue T = M^{-1}(P(.)^b)) so the affine mixing is paid in cheap
# CNOTs -- but then realize the residue T with TEXTBOOK out-of-place reversible
# synthesis (compute T into ancilla from its ANF, swap into place, uncompute with
# the ANF of T^{-1}). That doubles the residue Toffolis and adds a swap network,
# because it never notices the residue is triangular and can be done in place.
# Falls back to structure-blind synthesis if the peel is not clean.
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
    A = [rows[i] | (1 << (n + i)) for i in range(n)]
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
    return [(A[r] >> n) & ((1 << n) - 1) for r in range(n)]


def synth_linear_cnots(rows, n):
    A = rows[:]
    ops = []
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


def emit_anf(anf, n, ctrl_base, tgt_base, out):
    for j in range(n):
        tj = tgt_base + j
        for m in anf[j]:
            bs = bits_of(m)
            if not bs:
                out.append("NOT %d" % tj)
            elif len(bs) == 1:
                out.append("CNOT %d %d" % (ctrl_base + bs[0], tj))
            else:
                out.append("TOF %d %d %d" % (ctrl_base + bs[0], ctrl_base + bs[1], tj))


def blind_csu(P, n):
    Pinv = [0] * (1 << n)
    for x in range(1 << n):
        Pinv[P[x]] = x
    out = []
    emit_anf(anf_bits(P, n), n, 0, n, out)
    for i in range(n):
        out += ["CNOT %d %d" % (i, n + i), "CNOT %d %d" % (n + i, i), "CNOT %d %d" % (i, n + i)]
    emit_anf(anf_bits(Pinv, n), n, 0, n, out)
    return out


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); a = int(next(it))
    int(next(it)); int(next(it)); int(next(it))
    N = 1 << n
    P = [int(next(it)) for _ in range(N)]

    b = P[0]
    cols = [P[1 << i] ^ b for i in range(n)]
    rows = [0] * n
    for i in range(n):
        for j in range(n):
            if (cols[i] >> j) & 1:
                rows[j] |= (1 << i)
    Minv = invert_gf2(rows, n)
    if Minv is None:
        sys.stdout.write("\n".join(blind_csu(P, n)) + "\n"); return

    T = [matvec(Minv, P[x] ^ b, n) for x in range(N)]
    Tinv = [0] * N
    for x in range(N):
        Tinv[T[x]] = x

    out = []
    # compute-swap-uncompute the residue T (textbook, ancilla n..2n-1)
    emit_anf(anf_bits(T, n), n, 0, n, out)
    for i in range(n):
        out += ["CNOT %d %d" % (i, n + i), "CNOT %d %d" % (n + i, i), "CNOT %d %d" % (i, n + i)]
    emit_anf(anf_bits(Tinv, n), n, 0, n, out)
    # apply the peeled affine shell A(v) = M*v XOR b
    for (c, tg) in synth_linear_cnots(rows, n):
        out.append("CNOT %d %d" % (c, tg))
    for j in range(n):
        if (b >> j) & 1:
            out.append("NOT %d" % j)

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
