import os, sys, math
# pin BLAS to a single thread so the eigenvalue score is bit-for-bit reproducible.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def assemble(n, m, st):
    M = np.array(m, dtype=float)
    K = np.zeros((n, n))
    for j in range(n):
        K[j, j] += st[j]
        if j + 1 < n:
            K[j, j] += st[j + 1]
            K[j, j + 1] -= st[j + 1]
            K[j + 1, j] -= st[j + 1]
    return M, K


def decay_margin(n, m, st, beta, c_unit, t):
    # Assemble the 2n x 2n state matrix of  M x'' + C x' + K x = 0, where
    # C = beta*K (material damping) + diag(t_j * c_unit) (grounded viscous dampers),
    # then return the decay margin = -max Re(eigenvalue).  Higher = quieter.
    M, K = assemble(n, m, st)
    C = beta * K.copy()
    for j in range(n):
        C[j, j] += t[j] * c_unit
    Minv = np.diag(1.0 / M)
    A = np.zeros((2 * n, 2 * n))
    A[:n, n:] = np.eye(n)
    A[n:, :n] = -Minv @ K
    A[n:, n:] = -Minv @ C
    ev = np.linalg.eigvals(A)
    return -float(np.max(ev.real))


def antinode_baseline(n, m, st, K_total, T_max):
    # Internal baseline B: the naive "damp the biggest sway" placement -- pour all
    # units onto the floors with the largest fundamental-mode amplitude.  This is the
    # overdamping trap and is deterministic (tie-break by floor index).
    M, K = assemble(n, m, st)
    msi = 1.0 / np.sqrt(M)
    Ks = (msi[:, None] * K) * msi[None, :]
    w, U = np.linalg.eigh(0.5 * (Ks + Ks.T))
    phi1 = msi * U[:, 0]
    amp = phi1 * phi1
    order = sorted(range(n), key=lambda j: (-amp[j], j))
    t = [0] * n
    b = K_total
    for j in order:
        if b <= 0:
            break
        add = min(T_max, b)
        t[j] = add
        b -= add
    return t


def main():
    try:
        toks = open(sys.argv[1]).read().split()
        it = iter(toks)
        n = int(next(it)); K_total = int(next(it)); T_max = int(next(it))
        c_unit = float(next(it)); beta = float(next(it))
        m = []; st = []
        for _ in range(n):
            m.append(float(next(it))); st.append(float(next(it)))
    except Exception:
        fail("bad instance")

    out = open(sys.argv[2]).read().split()
    if len(out) != n:
        fail("expected %d integers, got %d" % (n, len(out)))
    t = []
    for tok in out:
        try:
            v = int(tok)
        except Exception:
            fail("non-integer token %r" % tok)
        if v < 0 or v > T_max:
            fail("unit count out of [0,%d]" % T_max)
        t.append(v)
    if sum(t) != K_total:
        fail("total units %d != required %d" % (sum(t), K_total))

    F = decay_margin(n, m, st, beta, c_unit, t)
    if not math.isfinite(F) or F <= 0:
        fail("non-positive/non-finite margin")

    tb = antinode_baseline(n, m, st, K_total, T_max)
    B = decay_margin(n, m, st, beta, c_unit, tb)
    B = max(1e-9, B)

    F = round(F, 9)
    B = round(B, 9)
    sc = min(1000.0, 100.0 * F / B)
    print("F=%.9f B=%.9f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
