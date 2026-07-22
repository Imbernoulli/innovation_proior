# TIER: greedy
"""Naive single-shot phase design (the obvious first attempt): average the two
target images in target space, treat sqrt(target) as the desired FOCAL
amplitude, take ONE inverse propagation (through plane A's operator only) and
read off the phase. This ignores that (a) the incident amplitude at the
aperture is fixed and uniform -- so the amplitude of the raw inverse-transform
result is thrown away, not actually achieved -- and (b) plane B has a
DIFFERENT propagation operator that this one-shot guess never consults. No
iteration, no feedback between the two constraint sets.
"""
import sys
import numpy as np


def _chirp(N, alpha, sign):
    c = (N - 1) / 2.0
    ii, jj = np.indices((N, N))
    return np.exp(sign * 1j * alpha * ((ii - c) ** 2 + (jj - c) ** 2))


def _backward(F, alpha):
    N = F.shape[0]
    field = np.fft.ifft2(F, norm="ortho")
    return field * _chirp(N, alpha, -1.0)


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it))
    aA = float(next(it))
    aB = float(next(it))  # read but never used -- that's the trap
    lam = float(next(it))
    TA = np.array([float(next(it)) for _ in range(N * N)]).reshape(N, N)
    TB = np.array([float(next(it)) for _ in range(N * N)]).reshape(N, N)

    Tavg = TA / TA.sum() + TB / TB.sum()
    Qamp = np.sqrt(Tavg / Tavg.sum() * (N * N))
    ap = _backward(Qamp.astype(complex), aA)
    theta = np.angle(ap)

    out = [str(N)]
    for i in range(N):
        out.append(" ".join("%.10f" % v for v in theta[i]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
