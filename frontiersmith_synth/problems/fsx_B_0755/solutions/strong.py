# TIER: strong
"""
The insight: the scored objective Dev = max over PASS/STOP bins of ||H(k)|-T_k|
is a SPECTRAL quantity, and the rounding error introduced by quantizing a real
design to integers is, itself, just another signal with a DFT. Pointwise
rounding (round each coefficient to its nearest integer, independently) treats
that error as noise to be minimised coefficient-by-coefficient in the TIME
domain -- but nothing says the error has to fall where it lands. Since DONTCARE
bins are never scored, the only thing that matters is where the error's energy
ends up in the FREQUENCY domain, not how large it is pointwise.

So instead of rounding once, this solution starts from the same "exact
real-valued match, DONTCARE pinned to 0" design the textbook approach uses,
then runs an integer-lattice coordinate search that is graded directly by the
true spectral objective: at every step it tries nudging every gear tooth by
+-1 (subject to the L1 budget) and keeps whichever single nudge shrinks the
worst-case PASS/STOP gap the most. A nudge that only perturbs DONTCARE-band
energy is invisible to this objective and is never preferred over one that
actually helps a scored bin -- so the search naturally discovers exactly the
"push the quantization error into the don't-care bins" trades that pointwise
rounding cannot see, without ever having to name them explicitly. It stops
when no single nudge helps anymore (a local optimum of the true objective,
not of a pointwise proxy for it).
"""
import sys
import numpy as np


def read_instance():
    toks = sys.stdin.read().split()
    idx = 0
    N = int(toks[idx]); idx += 1
    B = int(toks[idx]); idx += 1
    M = int(toks[idx]); idx += 1
    roles = []
    for _ in range(M):
        k = int(toks[idx]); idx += 1
        typ = toks[idx]; idx += 1
        T = float(toks[idx]); idx += 1
        roles.append((k, typ, T))
    return N, B, roles


def particular_solution(N, roles):
    """Exact real filter matching every PASS/STOP bin, DONTCARE pinned to 0 --
    the same starting point a textbook design would round directly."""
    C = np.zeros(N, dtype=complex)
    for k, typ, T in roles:
        val = T if typ == 'P' else 0.0
        C[k] = val
        if k != 0 and 2 * k != N:
            C[N - k] = val
    return np.fft.ifft(C).real


def repair(z, B):
    z = z.copy()
    guard = 0
    while int(np.sum(np.abs(z))) > B and guard < 10 * len(z):
        i = int(np.argmax(np.abs(z)))
        if z[i] > 0:
            z[i] -= 1
        elif z[i] < 0:
            z[i] += 1
        else:
            break
        guard += 1
    return z


def main():
    N, B, roles = read_instance()
    M = len(roles)
    ks = np.array([k for k, typ, T in roles])
    T_arr = np.array([T if typ == 'P' else 0.0 for k, typ, T in roles])
    mask_S = np.array([typ != 'D' for k, typ, T in roles])  # scored bins

    h_p = particular_solution(N, roles)
    z = repair(np.round(h_p).astype(np.int64), B)

    # E[j,k] = exp(-2*pi*i*j*k/N): coefficient j's contribution to bin k.
    j_idx = np.arange(N).reshape(N, 1)
    E = np.exp(-2j * np.pi * j_idx * ks.reshape(1, M) / N)

    H = (z.astype(complex) @ E)  # current response at every canonical bin

    def dev_of(Hvec):
        d = np.abs(np.abs(Hvec) - T_arr)
        d = np.where(mask_S, d, -np.inf)
        return float(np.max(d))

    current_dev = dev_of(H)
    current_l1 = int(np.sum(np.abs(z)))

    max_iters = 6 * N
    for _ in range(max_iters):
        H_plus = H[None, :] + E          # (N, M): result of z_j += 1
        H_minus = H[None, :] - E         # (N, M): result of z_j -= 1

        diff_plus = np.abs(np.abs(H_plus) - T_arr[None, :])
        diff_minus = np.abs(np.abs(H_minus) - T_arr[None, :])
        diff_plus = np.where(mask_S[None, :], diff_plus, -np.inf)
        diff_minus = np.where(mask_S[None, :], diff_minus, -np.inf)
        dev_plus = diff_plus.max(axis=1)
        dev_minus = diff_minus.max(axis=1)

        new_l1_plus = current_l1 - np.abs(z) + np.abs(z + 1)
        new_l1_minus = current_l1 - np.abs(z) + np.abs(z - 1)
        dev_plus = np.where(new_l1_plus <= B, dev_plus, np.inf)
        dev_minus = np.where(new_l1_minus <= B, dev_minus, np.inf)

        best_plus_j = int(np.argmin(dev_plus))
        best_minus_j = int(np.argmin(dev_minus))
        if dev_plus[best_plus_j] <= dev_minus[best_minus_j]:
            best_dev, best_j, best_delta = dev_plus[best_plus_j], best_plus_j, 1
        else:
            best_dev, best_j, best_delta = dev_minus[best_minus_j], best_minus_j, -1

        if best_dev >= current_dev - 1e-12:
            break  # no single nudge helps anymore -- local optimum reached

        z[best_j] += best_delta
        H = H + best_delta * E[best_j]
        current_l1 = current_l1 - abs(z[best_j] - best_delta) + abs(z[best_j])
        current_dev = best_dev

    z = repair(z, B)
    print(' '.join(str(int(x)) for x in z))


if __name__ == "__main__":
    main()
