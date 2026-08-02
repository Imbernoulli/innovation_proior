# TIER: strong
"""
The insight: for a FIXED cross-channel noise contribution Q_c from the other channels,
SNR_c(P) = P / (N_ASE + Q_c*P^2 + eta_c*P^3) is unimodal in P -- it rises (ASE-limited),
peaks, then falls (nonlinearity-limited). The peak solves
    N_ASE - Q_c*P^2 - 2*eta_c*P^3 = 0
(found here by bisection, not a fixed rule like "near the ceiling"). That peak power
maximises the tier channel c can reach GIVEN its neighbours' current power -- but Q_c
itself depends on every other channel's power, so this is solved as coordinate ascent
(Gauss-Seidel sweeps over channels) rather than once per channel in isolation.

A second insight on top of the interior optimum: once channel c's best-reachable tier is
known, using MORE power than the least amount that still clears that tier's SNR
threshold only pumps extra, useless cross-channel noise onto every other channel. So
after finding the peak (and its tier), each channel is pulled back down to the SMALLEST
power on the monotonic-increasing branch [0, peak] that still holds the same tier --
this reduces collateral damage and, over further sweeps, lets other channels reach
higher tiers than a "just sit at your own peak" strategy would. Both refinements
(interior-optimum power AND jointly-aware footprint minimisation) are re-applied over
several sweeps to reach a good fixed point.
"""
import sys

ROUNDS = 6
BISECT_ITERS = 70


def bisect_decreasing_root(f, lo, hi, iters=BISECT_ITERS):
    """f(lo) >= 0 >= f(hi) (f non-increasing); returns approx root."""
    flo = f(lo)
    if flo < 0:
        return lo
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        fm = f(mid)
        if fm >= 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def bisect_smallest_ge(f, lo, hi, iters=BISECT_ITERS):
    """Smallest x in [lo,hi] with f(x) >= 0, given f non-decreasing on [lo,hi]."""
    if f(lo) >= 0:
        return lo
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return hi


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

    # init: everyone at their isolated (Q=0) peak, clipped to Pmax
    power = []
    for c in range(C):
        p0 = (n_ase / (2.0 * eta[c])) ** (1.0 / 3.0)
        power.append(min(pmax[c], p0))
    tier = [0] * C

    def q_of(c):
        qc = 0.0
        for c2 in range(C):
            if c2 == c:
                continue
            qc += kappa[c][c2] * power[c2] * power[c2]
        return qc

    def snr_at(c, pc, qc):
        if pc <= 0.0:
            return 0.0
        nli = eta[c] * pc ** 3 + qc * pc * pc
        return pc / (n_ase + nli)

    for _round in range(ROUNDS):
        for c in range(C):
            qc = q_of(c)
            ec = eta[c]
            pm = pmax[c]

            def h(P, qc=qc, ec=ec):
                return n_ase - qc * P * P - 2.0 * ec * P ** 3

            peak = bisect_decreasing_root(h, 0.0, pm)
            snr_peak = snr_at(c, peak, qc)

            best_m = 0
            for k in range(K):
                if snr_peak >= req[k] * (1.0 + 1e-9):
                    best_m = k + 1

            if best_m == 0:
                power[c] = 0.0
                tier[c] = 0
                continue

            target = req[best_m - 1] * (1.0 + 1e-4)

            def g(P, qc=qc, ec=ec, target=target, c=c):
                return snr_at(c, P, qc) - target

            p_lo = bisect_smallest_ge(g, 0.0, peak)
            power[c] = p_lo
            tier[c] = best_m

    # final verification pass at the converged power vector: recompute true SNR with
    # cross terms from the FINAL profile and re-derive the best safely-achievable tier
    # for each channel (guards against any residual drift from the sweeps above).
    lines = []
    for c in range(C):
        qc = q_of(c)
        s = snr_at(c, power[c], qc)
        m = 0
        for k in range(K):
            if s >= req[k] * (1.0 + 1e-6):
                m = k + 1
        if m == 0:
            lines.append("0.0 0")
        else:
            lines.append("%.9f %d" % (power[c], m))

    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
