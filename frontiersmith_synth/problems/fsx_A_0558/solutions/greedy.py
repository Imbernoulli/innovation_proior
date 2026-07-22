# TIER: greedy
# Obvious approach: FFT the target ENVELOPE E, place beat pairs at its dominant spacings
# matched in phase, and set the overall loudness so the flat level equals mean(E).
# Trap: it matches harmonics of E (amplitude domain) with near-uniform beat strength and
# ignores that the sliding-RMS lives in the POWER domain (E^2) and that the window
# attenuates fast beats -- so its shimmer is the wrong shape/strength.
import sys, numpy as np

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    N = int(next(it)); W = int(next(it))
    F_min = int(next(it)); F_max = int(next(it))
    k = int(next(it)); na = int(next(it))
    allowed = sorted(float(next(it)) for _ in range(na))
    E = np.array([float(next(it)) for _ in range(N)], dtype=float)

    T = max(1, k // 2)
    fftE = np.fft.rfft(E)
    gmax = min(18, len(fftE) - 1)
    mags = np.abs(fftE[1:gmax + 1])
    order = np.argsort(mags)[::-1]
    gsel = [int(1 + order[h]) for h in range(min(T, len(order)))]
    phase = {g: float(np.angle(fftE[g])) for g in gsel}
    Tsel = len(gsel)

    # naive level: choose amplitudes so the flat power level matches mean(E)^2.
    target_sumsq = 2.0 * (E.mean() ** 2)                 # sum a_j^2 (naive, uses mean E not mean E^2)
    ntone = 2 * Tsel
    base = min(allowed, key=lambda x: abs(x - np.sqrt(target_sumsq / ntone)))
    amps = [base] * ntone
    # small fill so the discrete sum-of-squares lands close to the target
    def sumsq(v): return sum(a * a for a in v)
    for _ in range(ntone):
        cur = abs(sumsq(amps) - target_sumsq)
        bestj, besta, bestd = -1, None, cur
        for j in range(ntone):
            for a in allowed:
                if a == amps[j]:
                    continue
                dd = abs(sumsq(amps[:j] + [a] + amps[j + 1:]) - target_sumsq)
                if dd < bestd - 1e-12:
                    bestd, bestj, besta = dd, j, a
        if bestj < 0:
            break
        amps[bestj] = besta

    lo = F_min + 4
    hi = F_max - gmax - 4
    carriers = np.linspace(lo, hi, Tsel)
    lines = []
    for h, g in enumerate(gsel):
        c = int(round(carriers[h]))
        lines.append("%d %.6f %.6f" % (c, amps[2 * h], 0.0))
        lines.append("%d %.6f %.6f" % (c + g, amps[2 * h + 1], phase[g]))
    sys.stdout.write("\n".join(lines) + "\n")

main()
