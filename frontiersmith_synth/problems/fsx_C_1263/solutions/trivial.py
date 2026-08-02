# TIER: trivial
"""
Mirrors the checker's own DELIBERATELY WEAK reference construction: for every channel,
spend just enough power to clear tier 1's threshold pretending no other channel exists
(self-noise only -- ignores the whole cross-channel-nonlinearity mechanism), then use
tier 1 and never bother checking whether a higher tier is reachable. Found by bisection
on the self-only SNR curve. This is the "do the least possible thing that works" baseline
-- it should score ~0.1.
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
    for _ in range(C):
        for _ in range(C):
            nx()  # kappa -- deliberately ignored

    n_ase = float(sum(ase))

    def self_snr(pc, c):
        if pc <= 0.0:
            return 0.0
        return pc / (n_ase + eta[c] * pc ** 3)

    lines = []
    for c in range(C):
        p0 = (n_ase / (2.0 * eta[c])) ** (1.0 / 3.0)
        # 25% margin above req[0]: the bisection below is self-only, but the checker
        # validates against the REAL SNR (which includes whatever small cross-channel
        # noise this simultaneous low-power profile actually produces), so aim above
        # the bare threshold rather than exactly at it.
        target = req[0] * 1.25
        if self_snr(p0, c) < target:
            lines.append("0.0 0")
            continue
        lo, hi = 0.0, p0
        for _ in range(80):
            mid = (lo + hi) / 2.0
            if self_snr(mid, c) < target:
                lo = mid
            else:
                hi = mid
        power_c = min(hi, pmax[c])
        lines.append("%.9f 1" % power_c)

    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
