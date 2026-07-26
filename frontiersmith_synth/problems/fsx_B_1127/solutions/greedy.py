# TIER: greedy
"""
The obvious "average strong coder" move: relaxation curves are classically
fit with a sum of exponentials (a Prony / generalised-Maxwell series). Fix a
few relaxation times spanning the observed window and least-squares fit the
amplitudes. This nails the TRAINING window (it's a flexible linear-in-
amplitude model with 3 free knobs) but each exponential has its OWN
frequency-dependent phase response, so extrapolated far outside the fitted
decades (much faster or much slower straining) the prediction drifts badly
-- it never discovers that the true loss tangent should stay frequency-
independent. More terms would fit the window even better without fixing
that.
"""
import sys
import math

try:
    import numpy as np
except Exception:
    np = None


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    rows = []
    idx = 2
    for _ in range(n):
        g0 = float(data[idx]); tt = float(data[idx + 1]); sig = float(data[idx + 2])
        rows.append((g0, tt, sig))
        idx += 3

    ts = [r[1] for r in rows]
    t_min, t_max = min(ts), max(ts)
    k = 3
    taus = [t_min * (t_max / t_min) ** (i / (k - 1)) for i in range(k)]

    xs = [[math.exp(-t / tau) for tau in taus] for _, t, _ in rows]
    ys = [s / g0 for g0, _, s in rows]

    if np is not None:
        coef, *_ = np.linalg.lstsq(np.array(xs), np.array(ys), rcond=None)
        coef = [float(c) for c in coef]
    else:
        # plain normal-equations fallback (k is tiny, 3x3 solve by hand)
        AT_A = [[sum(xs[r][i] * xs[r][j] for r in range(len(xs))) for j in range(k)] for i in range(k)]
        AT_y = [sum(xs[r][i] * ys[r] for r in range(len(xs))) for i in range(k)]
        # Gaussian elimination
        M = [row[:] + [AT_y[i]] for i, row in enumerate(AT_A)]
        for c in range(k):
            piv = max(range(c, k), key=lambda r: abs(M[r][c]))
            M[c], M[piv] = M[piv], M[c]
            pv = M[c][c] if abs(M[c][c]) > 1e-12 else 1e-12
            for r in range(k):
                if r == c:
                    continue
                f = M[r][c] / pv
                for cc in range(k + 1):
                    M[r][cc] -= f * M[c][cc]
        coef = [M[i][k] / (M[i][i] if abs(M[i][i]) > 1e-12 else 1e-12) for i in range(k)]

    terms = ["( %.8f ) * exp( -t / %.8f )" % (coef[i], taus[i]) for i in range(k)]
    print(" + ".join(terms))


if __name__ == "__main__":
    main()
