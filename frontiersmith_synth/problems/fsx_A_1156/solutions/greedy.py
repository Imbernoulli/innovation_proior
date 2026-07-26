# TIER: greedy
# The obvious recipe: generic 4-sinusoid spectral fitting by successive single-
# frequency extraction ("matching pursuit" / CLEAN). Search the whole plausible
# band for the single best-fitting sinusoid, subtract it, repeat four times,
# each frequency refined independently by a local grid search. This reproduces
# the training log closely -- but the three LOCKED gears sit closer together in
# frequency than a window this short can resolve, so treating all four
# frequencies as independent unknowns is an ill-posed fit: many different
# frequency quadruples explain the training window almost equally well, and the
# one this recipe lands on is essentially arbitrary among them. Extrapolated to
# the held-out window (several locked super-periods later) that arbitrary
# frequency error blows up into an uncorrelated phase error. The recipe never
# asks whether three of its four frequencies secretly share an exact small-
# integer ratio.
import sys, math
import numpy as np


def fit_amp_phase(t, y, w):
    C = np.cos(w * t); S = np.sin(w * t)
    M = np.column_stack([C, S])
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    a, b = coef
    A = math.hypot(a, b)
    phi = math.atan2(a, b)
    resid = y - (a * C + b * S)
    return A, phi, resid


def best_freq(t, y, flo, fhi, ngrid):
    fs = np.linspace(flo, fhi, ngrid)
    best = None
    for f in fs:
        C = np.cos(2 * math.pi * f * t); S = np.sin(2 * math.pi * f * t)
        M = np.column_stack([C, S])
        coef, *_ = np.linalg.lstsq(M, y, rcond=None)
        pred = M @ coef
        rss = float(np.sum((y - pred) ** 2))
        if best is None or rss < best[0]:
            best = (rss, f)
    return best[1]


def refine(t, y, f0, flo, fhi, span, rounds, ngrid):
    lo, hi = max(flo, f0 - span), min(fhi, f0 + span)
    for _ in range(rounds):
        f0 = best_freq(t, y, lo, hi, ngrid)
        span = span / 8.0
        lo, hi = max(flo, f0 - span), min(fhi, f0 + span)
    return f0


def matching_pursuit(t, y, flo=0.003, fhi=0.32, ngrid=700, n_components=4):
    resid = y.copy()
    comps = []
    for _ in range(n_components):
        f0 = best_freq(t, resid, flo, fhi, ngrid)
        span = (fhi - flo) / ngrid * 6
        f0 = refine(t, resid, f0, flo, fhi, span, 3, 200)
        A, phi, resid = fit_amp_phase(t, resid, 2 * math.pi * f0)
        comps.append((A, f0, phi))
    return comps


def main():
    data = sys.stdin.read().split()
    if not data:
        print("0.0"); return
    n = int(data[0])
    vals = data[2:]
    t = np.array([float(vals[2 * i]) for i in range(n)])
    y = np.array([float(vals[2 * i + 1]) for i in range(n)])

    comps = matching_pursuit(t, y)
    terms = []
    for A, f, phi in comps:
        w = 2 * math.pi * f
        terms.append("%.10g * sin ( %.10g * t + %.10g )" % (A, w, phi))
    print(" + ".join(terms))


if __name__ == "__main__":
    main()
