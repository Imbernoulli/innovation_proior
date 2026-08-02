# TIER: greedy
"""
The obvious "more power is more signal" recipe: launch EVERY channel at its declared
ceiling Pmax_c, then -- honestly, using the real formula -- compute the resulting SNR
(including the cross-channel term, since all channels are simultaneously at their
ceiling this is a single well-defined pass, no iteration needed) and pick the highest
modulation tier that SNR actually supports (0 / silent if none). It never considers
backing off: it does not model that fibre-nonlinearity noise grows faster than linearly
with power, and it never asks whether a lower power would let it (or a neighbour) reach
a HIGHER tier. On short low-noise instances the ceiling is close to the true optimum, so
this is fine; on long-haul instances, where the ceiling is set well above every channel's
true SNR peak, blasting every channel at once collapses many channels' SNR far below
what a tuned power would achieve.
"""
import sys


def main():
    data = sys.stdin.read().split()
    p = iter(data)

    def nx():
        return next(p)

    C = int(nx())
    S = int(nx())
    ase = [int(nx()) for _ in range(S)]
    eta = [float(nx()) for _ in range(C)]
    K = int(nx())
    bps = []
    req = []
    for _ in range(K):
        bps.append(int(nx()))
        req.append(float(nx()))
    pmax = [float(nx()) for _ in range(C)]
    baud = float(nx())
    kappa = [[float(nx()) for _ in range(C)] for _ in range(C)]

    n_ase = float(sum(ase))

    power = list(pmax)  # max out every channel

    def snr_of(c):
        pc = power[c]
        if pc <= 0.0:
            return 0.0
        qc = 0.0
        for c2 in range(C):
            if c2 == c:
                continue
            qc += kappa[c][c2] * power[c2] * power[c2]
        nli = eta[c] * pc ** 3 + qc * pc * pc
        return pc / (n_ase + nli)

    lines = []
    for c in range(C):
        s = snr_of(c)
        m = 0
        for k in range(K):
            if s >= req[k]:
                m = k + 1
        if m == 0:
            lines.append("0.0 0")
        else:
            lines.append("%.9f %d" % (power[c], m))

    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
