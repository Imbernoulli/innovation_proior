#!/usr/bin/env python3
# Generator for "Thin-Film Coating: Hit a Reflectance Spectrum" (format C, minimize).
# `python3 gen.py <testId>` prints ONE instance to stdout. Deterministic in testId only.
#
# TARGET: the reflectance spectrum Rstar(lambda) of a HIDDEN reference multilayer that has
# MANY MORE films than the solver's budget L (10-12 films vs L=3-4). Its refractive-index
# pattern and thicknesses are pseudo-random (seeded by the test), so the spectrum is a rich,
# non-periodic curve. No L-film stack of the 3 available materials can reproduce a 10-12
# film spectrum, so the best design is forced to leave a real residual -> the score never
# saturates and the true optimum is unknown.
#
# Why it is a trap: a single quarter-wave film nulls the mismatch at ONE wavelength but
# disturbs all the others. Naively "adding a quarter-wave to null the worst wavelength" each
# time (the obvious greedy) thrashes, because every film couples all wavelengths at once. A
# strong design instead reasons about how the whole stack moves optical ADMITTANCE across
# the WHOLE band at once, so it tracks the target curve far better with the same few films.
import sys, math

N0 = 1.0


def reflectance(layers, ns, lam):
    m00, m01, m10, m11 = 1 + 0j, 0j, 0j, 1 + 0j
    for (n, d) in layers:
        delta = 2.0 * math.pi * n * d / lam
        c = math.cos(delta); s = math.sin(delta)
        a00, a01, a10, a11 = c, 1j * s / n, 1j * n * s, c
        m00, m01, m10, m11 = (m00 * a00 + m01 * a10, m00 * a01 + m01 * a11,
                              m10 * a00 + m11 * a10, m10 * a01 + m11 * a11)
    B = m00 * 1.0 + m01 * ns
    C = m10 * 1.0 + m11 * ns
    Y = C / B
    r = (N0 - Y) / (N0 + Y)
    R = r.real * r.real + r.imag * r.imag
    return min(1.0, max(0.0, R))


def make_reference(mats, band, reflen, seed):
    # Deterministic LCG -> a rich non-periodic reference stack (materials + thicknesses).
    lmin, lmax = band
    rng = (1103515245 * seed + 12345) & 0x7fffffff
    layers = []
    for _ in range(reflen):
        rng = (1103515245 * rng + 12345) & 0x7fffffff
        mi = (rng >> 5) % len(mats)
        n = mats[mi]
        rng = (1103515245 * rng + 12345) & 0x7fffffff
        lw = lmin + (lmax - lmin) * ((rng % 1000) / 1000.0)
        layers.append((n, lw / (4.0 * n)))
    return layers


# Per-test config: (ns, mats, band, K, reflen, L, kappa, seed). Selected so each test lands
# strong-solution ratio in ~[0.61,0.86], greedy well below it, greedy above the do-nothing
# baseline. Difficulty rises with testId (larger substrate mismatch / more films needed).
TESTS = [
    dict(ns=2.1, mats=[1.46, 1.85, 2.30], band=(420, 800), K=13, reflen=12, L=3, kappa=0.05, seed=2),
    dict(ns=2.0, mats=[1.38, 1.80, 2.30], band=(420, 780), K=13, reflen=11, L=3, kappa=0.05, seed=45),
    dict(ns=2.1, mats=[1.46, 1.85, 2.30], band=(420, 800), K=13, reflen=12, L=3, kappa=0.05, seed=48),
    dict(ns=1.9, mats=[1.38, 1.75, 2.20], band=(430, 760), K=15, reflen=10, L=4, kappa=0.05, seed=41),
    dict(ns=2.2, mats=[1.46, 1.90, 2.35], band=(410, 820), K=15, reflen=12, L=4, kappa=0.05, seed=2),
    dict(ns=2.3, mats=[1.46, 1.90, 2.35], band=(410, 820), K=15, reflen=12, L=4, kappa=0.05, seed=2),
    dict(ns=2.2, mats=[1.46, 1.90, 2.35], band=(410, 820), K=15, reflen=12, L=4, kappa=0.05, seed=7),  # TRAP
    dict(ns=2.3, mats=[1.46, 1.90, 2.35], band=(410, 820), K=15, reflen=12, L=4, kappa=0.05, seed=7),  # TRAP
    dict(ns=1.9, mats=[1.38, 1.75, 2.20], band=(430, 760), K=15, reflen=10, L=4, kappa=0.05, seed=10),
    dict(ns=2.2, mats=[1.46, 1.90, 2.35], band=(410, 820), K=15, reflen=12, L=4, kappa=0.05, seed=32),  # TRAP
]


def main():
    tid = int(sys.argv[1])
    cfg = TESTS[min(max(tid, 1), len(TESTS)) - 1]
    ns = cfg["ns"]; mats = cfg["mats"]; K = cfg["K"]
    lmin, lmax = cfg["band"]; L = cfg["L"]
    lam0 = 0.5 * (lmin + lmax)
    dmax = 1200.0

    lams = [lmin + (lmax - lmin) * k / (K - 1) for k in range(K)]
    ref = make_reference(mats, cfg["band"], cfg["reflen"], cfg["seed"])
    Rstar = [reflectance(ref, ns, l) for l in lams]

    Rbare = ((N0 - ns) / (N0 + ns)) ** 2
    B = sum((Rbare - r) ** 2 for r in Rstar)
    cost = cfg["kappa"] * B / L

    out = []
    out.append("%.6f %.6f" % (N0, ns))
    out.append(str(len(mats)))
    out.append(" ".join("%.4f" % n for n in mats))
    out.append(str(K))
    for lam, r in zip(lams, Rstar):
        out.append("%.4f %.8f" % (lam, r))
    out.append("%d %.4f %.10f %.4f" % (L, lam0, cost, dmax))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
