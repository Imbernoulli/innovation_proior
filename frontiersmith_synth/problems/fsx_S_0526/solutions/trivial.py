# TIER: trivial
# Structure-blind compute-swap-uncompute over the FULL table's Reed-Muller ANF.
# For EVERY output bit it recomputes each residue monomial from scratch (no
# sharing), realizes y=P(x) in ancilla, swaps into place, then uncomputes the
# ancilla with the ANF of P^{-1}. This reproduces the checker's baseline B -> ~0.1.
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


def emit_anf(anf, n, ctrl_base, tgt_base, out):
    # ctrl_base: base index of the SOURCE (input) wires; tgt_base: base of target
    # wires. Writes each output bit's ANF into wire tgt_base+j (assumed 0).
    for j in range(n):
        tj = tgt_base + j
        for m in anf[j]:
            bs = bits_of(m)
            if len(bs) == 0:
                out.append("NOT %d" % tj)
            elif len(bs) == 1:
                out.append("CNOT %d %d" % (ctrl_base + bs[0], tj))
            else:
                out.append("TOF %d %d %d" % (ctrl_base + bs[0], ctrl_base + bs[1], tj))


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); a = int(next(it))
    int(next(it)); int(next(it)); int(next(it))  # costs (unused here)
    N = 1 << n
    P = [int(next(it)) for _ in range(N)]
    Pinv = [0] * N
    for x in range(N):
        Pinv[P[x]] = x

    anfP = anf_bits(P, n)
    anfPi = anf_bits(Pinv, n)
    out = []

    # compute y = P(x) into ancilla wires n..2n-1 (controls = data wires 0..n-1)
    emit_anf(anfP, n, 0, n, out)
    # swap data<->y  (3 CNOTs per bit)
    for i in range(n):
        out.append("CNOT %d %d" % (i, n + i))
        out.append("CNOT %d %d" % (n + i, i))
        out.append("CNOT %d %d" % (i, n + i))
    # uncompute ancilla (=x) using P^{-1}(register) : ancilla_i ^= Pinv_i(data)
    emit_anf(anfPi, n, 0, n, out)

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
