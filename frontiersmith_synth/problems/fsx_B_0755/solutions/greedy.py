# TIER: greedy
"""
The textbook first attempt: design the exact real-valued (continuous-coefficient)
circulant filter that hits every PASS bin at its target gain and drives every
STOP *and* DONTCARE bin to zero (a very natural "if it isn't asked for, keep it
quiet" instinct -- a smooth, conservative design). Scale the result down (if
needed) to respect the L1 budget, round every coefficient to the nearest
integer, then repair any post-rounding budget overshoot by shaving the
largest-magnitude coefficients toward zero.

This treats rounding as pointwise noise to be minimized coefficient-by-
coefficient, and spends part of the limited L1 budget shaping bins that are
never scored -- both mistakes that a genuinely open-ended reformulation avoids.
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


def main():
    N, B, roles = read_instance()

    C = np.zeros(N, dtype=complex)
    for k, typ, T in roles:
        val = T if typ == 'P' else 0.0  # STOP and DONTCARE both forced to 0
        C[k] = val
        if k != 0 and 2 * k != N:
            C[N - k] = val

    h_real = np.fft.ifft(C).real
    l1 = float(np.sum(np.abs(h_real)))
    scale = min(1.0, B / l1) if l1 > 1e-9 else 1.0
    h_int = np.round(h_real * scale).astype(int)

    def l1n(v):
        return int(np.sum(np.abs(v)))

    guard = 0
    while l1n(h_int) > B and guard < 10 * N:
        i = int(np.argmax(np.abs(h_int)))
        if h_int[i] > 0:
            h_int[i] -= 1
        elif h_int[i] < 0:
            h_int[i] += 1
        else:
            break
        guard += 1

    print(' '.join(str(int(x)) for x in h_int))


if __name__ == "__main__":
    main()
