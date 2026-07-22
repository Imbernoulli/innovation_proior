# TIER: strong
# INSIGHT: the sliding-RMS is exact only in DIFFERENCE-FREQUENCY / POWER space.
# power[n] = (1/2) sum a_j^2  +  sum_{i<j, |f_i-f_j| small} a_i a_j cos(2pi(f_i-f_j)n/N + dphi)
# So RMS^2 is LINEAR in the pairwise spacings.  Design in E^2 (not E):
#   * decompose P = E^2 into DC (P0) + slow cosines c_g cos(2pi g n/N + psi_g),
#   * realize each chosen harmonic g with a disjoint pair (carrier, carrier+g) whose BEAT
#     reproduces that cosine, and set the phase difference = psi_g,
#   * fit the discrete amplitudes by minimizing the TRUE envelope error of this skeleton.
import sys, numpy as np

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    N = int(next(it)); W = int(next(it))
    F_min = int(next(it)); F_max = int(next(it))
    k = int(next(it)); na = int(next(it))
    allowed = sorted(float(next(it)) for _ in range(na))
    E = np.array([float(next(it)) for _ in range(N)], dtype=float)

    n = np.arange(N)
    half = (W - 1) // 2

    def env_of(freqs, amps, phis):
        s = np.zeros(N)
        for f, a, p in zip(freqs, amps, phis):
            s += a * np.cos(2 * np.pi * f * n / N + p)
        s2 = s * s
        ext = np.concatenate([s2[N - half:], s2, s2[:half]])
        cs = np.concatenate([[0.0], np.cumsum(ext)])
        win = cs[W:W + N] - cs[0:N]
        return np.sqrt(np.clip(win / W, 0.0, None))

    def err_of(freqs, amps, phis):
        return float(np.sqrt(np.mean((env_of(freqs, amps, phis) - E) ** 2)))

    T = max(1, k // 2)
    P = E * E
    fftP = np.fft.rfft(P)
    fftE = np.fft.rfft(E)
    gmax = min(18, len(fftP) - 1)
    dc0 = float(fftP[0].real / N)                       # mean(E^2) = correct power DC

    def top_harmonics(spec):
        cand = []
        for g in range(1, gmax + 1):
            w = np.sin(np.pi * g * W / N) / (W * np.sin(np.pi * g / N))
            if w < 0.15:
                continue
            cand.append((abs(spec[g]), g, float(np.angle(spec[g]))))
        cand.sort(reverse=True)
        return cand[:T]

    lo = F_min + 4
    hi = F_max - gmax - 4

    def descend(freqs, phis, amps):
        cur = err_of(freqs, amps, phis)
        for _ in range(6):
            improved = False
            for j in range(len(amps)):
                best_a, best_e = amps[j], cur
                for a in allowed:
                    if a == amps[j]:
                        continue
                    amps[j] = a
                    e = err_of(freqs, amps, phis)
                    if e < best_e - 1e-12:
                        best_e, best_a = e, a
                amps[j] = best_a
                if best_e < cur - 1e-12:
                    cur = best_e; improved = True
            if not improved:
                break
        return amps, cur

    def build_and_fit(cand):
        Tsel = len(cand)
        if Tsel == 0:
            return None, 1e18
        carriers = [int(round(x)) for x in np.linspace(lo, hi, Tsel)]
        freqs, phis = [], []
        for h, (_m, g, psi) in enumerate(cand):
            freqs += [carriers[h], carriers[h] + g]
            phis  += [0.0, psi]
        # multi-start coordinate descent (uniform starts avoid the asymmetric local traps)
        starts = [np.sqrt(dc0 / (2 * Tsel))] + list(allowed)
        best_amps, best_e = None, 1e18
        seen = set()
        for sv in starts:
            a0 = min(allowed, key=lambda x: abs(x - sv))
            if a0 in seen:
                continue
            seen.add(a0)
            amps, e = descend(freqs, phis, [a0] * (2 * Tsel))
            if e < best_e:
                best_e, best_amps = e, amps
        return (freqs, list(best_amps), phis), best_e

    # try the POWER-domain (E^2) skeleton and, as a safety floor, the envelope (E) skeleton;
    # amplitudes are fit against the true model, so this dominates the naive recipe.
    best = None; best_e = 1e18
    for spec in (fftP, fftE):
        design, e = build_and_fit(top_harmonics(spec))
        if design is not None and e < best_e:
            best_e, best = e, design

    if best is None:
        a = min(allowed, key=lambda x: abs(x - np.sqrt(dc0)))
        print("%d %.6f %.6f" % (F_min, a, 0.0)); return

    freqs, amps, phis = best
    lines = ["%d %.6f %.6f" % (freqs[j], amps[j], phis[j]) for j in range(len(freqs))]
    sys.stdout.write("\n".join(lines) + "\n")

main()
