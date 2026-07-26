# TIER: strong
# The insight: this is a discrete MODEL-SELECTION problem hiding inside a
# spectral fit, not a 4-parameter frequency-fitting problem. The training
# window only spans part of one locked super-period, so the three locked
# gears are NOT independently resolvable -- but the historical harbor charts
# only admit a short menu of small-integer gear ratios. Instead of fitting
# four unrelated frequencies, try each candidate integer triple (n1,n2,n3) and,
# for each one, fit a SINGLE shared base rate f0 jointly against all three
# harmonics at once (a 6-parameter LINEAR fit given f0, searched over a 1-D
# grid in f0). Using three harmonics -- especially the fast one, n3*f0 --
# pins down f0 far more precisely than any lone frequency could be resolved
# from this short window, and because the multipliers are forced to be EXACT
# integers there is no per-component ratio error to dephase later. Pick the
# candidate triple with the lowest residual, then fit the leftover free
# interloper frequency on what remains. The locked components then stay phase
# -coherent all the way to the held-out window, several super-periods later,
# where the greedy per-frequency fit has already lost coherence.
import sys, math
import numpy as np

CANDIDATE_TRIPLES = [(2, 3, 7), (3, 4, 5), (2, 5, 7), (3, 5, 8),
                      (2, 3, 11), (4, 5, 7), (2, 7, 9), (3, 7, 8)]
F0_LO, F0_HI = 0.0045, 0.0065


def locked_fit(t, y, n1, n2, n3, f0):
    ws = (2 * math.pi * n1 * f0, 2 * math.pi * n2 * f0, 2 * math.pi * n3 * f0)
    cols = []
    for w in ws:
        cols.append(np.cos(w * t)); cols.append(np.sin(w * t))
    M = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    pred = M @ coef
    rss = float(np.sum((y - pred) ** 2))
    return rss, coef, pred


def search_f0(t, y, n1, n2, n3, lo, hi, ngrid):
    fs = np.linspace(lo, hi, ngrid)
    best = None
    for f0 in fs:
        rss, coef, pred = locked_fit(t, y, n1, n2, n3, f0)
        if best is None or rss < best[0]:
            best = (rss, f0, coef)
    return best


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


def fit_amp_phase(t, y, w):
    C = np.cos(w * t); S = np.sin(w * t)
    M = np.column_stack([C, S])
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    a, b = coef
    return math.hypot(a, b), math.atan2(a, b)


def main():
    data = sys.stdin.read().split()
    if not data:
        print("0.0"); return
    n = int(data[0])
    vals = data[2:]
    t = np.array([float(vals[2 * i]) for i in range(n)])
    y = np.array([float(vals[2 * i + 1]) for i in range(n)])

    best_overall = None
    for (n1, n2, n3) in CANDIDATE_TRIPLES:
        rss, f0, coef = search_f0(t, y, n1, n2, n3, F0_LO, F0_HI, 900)
        span = (F0_HI - F0_LO) / 900 * 6
        lo, hi = max(F0_LO, f0 - span), min(F0_HI, f0 + span)
        for _ in range(4):
            rss, f0, coef = search_f0(t, y, n1, n2, n3, lo, hi, 400)
            span = span / 8.0
            lo, hi = max(F0_LO, f0 - span), min(F0_HI, f0 + span)
        if best_overall is None or rss < best_overall[0]:
            best_overall = (rss, n1, n2, n3, f0, coef)
    rss, n1, n2, n3, f0, coef = best_overall

    terms = []
    ns = (n1, n2, n3)
    for i, nk in enumerate(ns):
        a, b = coef[2 * i], coef[2 * i + 1]
        A = math.hypot(a, b); phi = math.atan2(a, b)
        w = 2 * math.pi * nk * f0
        terms.append("%.10g * sin ( %.10g * t + %.10g )" % (A, w, phi))

    _, _, pred = locked_fit(t, y, n1, n2, n3, f0)
    resid = y - pred

    f4 = best_freq(t, resid, 0.05, 0.32, 1200)
    span = (0.32 - 0.05) / 1200 * 6
    lo, hi = max(0.001, f4 - span), f4 + span
    for _ in range(3):
        f4 = best_freq(t, resid, lo, hi, 300)
        span = span / 8.0
        lo, hi = max(0.001, f4 - span), f4 + span
    A4, phi4 = fit_amp_phase(t, resid, 2 * math.pi * f4)
    terms.append("%.10g * sin ( %.10g * t + %.10g )" % (A4, 2 * math.pi * f4, phi4))

    print(" + ".join(terms))


if __name__ == "__main__":
    main()
