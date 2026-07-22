# TIER: greedy
# The obvious recipe: this "looks like" a single driven-damped-oscillator
# resonance-fitting problem, so collapse the lock-in data to a scalar
# amplitude (THROW AWAY the phase / quadrature channel entirely) and fit the
# textbook one-pole-pair Lorentzian
#     A(w) = c / sqrt((w0^2-w^2)^2 + (gamma0*w)^2)
# by bounded nonlinear least squares directly against the training tail
# (multi-start, deterministic seed). This matches the smooth, featureless
# tail curve well -- any locally-analytic curve is well approximated by one
# resonance shape over a narrow enough window. But a single Lorentzian
# structurally has only ONE resonance: extrapolated it predicts one peak and
# a 1/w^2 high-frequency roll-off, where the true box holds two coupled
# oscillators with two split peaks (an avoided crossing) and a 1/w^4
# roll-off. The recipe never even looks for a second mode -- it only ever
# had three numbers to spend.
import sys, math, random

try:
    from scipy.optimize import least_squares
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def model(params, w):
    c, w0sq, gamma0 = params
    denom = (w0sq - w * w) ** 2 + (gamma0 * w) ** 2
    return c / math.sqrt(denom) if denom > 1e-300 else 1e150


def residuals(params, ws, amps):
    return [model(params, w) - a for w, a in zip(ws, amps)]


def crude_fit(ws, amps):
    """Closed-form fallback: invert y=1/A^2, which is quadratic in u=w^2."""
    us = [w * w for w in ws]
    ys = [1.0 / max(a, 1e-9) ** 2 for a in amps]

    def solve(A, b):
        n = len(A)
        M = [row[:] + [b[i]] for i, row in enumerate(A)]
        for c in range(n):
            piv = max(range(c, n), key=lambda r: abs(M[r][c]))
            M[c], M[piv] = M[piv], M[c]
            d = M[c][c]
            if abs(d) < 1e-18:
                d = 1e-18
            for r in range(n):
                if r == c:
                    continue
                f = M[r][c] / d
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
        return [M[i][n] / (M[i][i] if abs(M[i][i]) > 1e-18 else 1e-18) for i in range(n)]

    k = 3
    A = [[0.0] * k for _ in range(k)]
    b = [0.0] * k
    for u, y in zip(us, ys):
        feat = [u * u, u, 1.0]
        for r in range(k):
            b[r] += feat[r] * y
            for c in range(k):
                A[r][c] += feat[r] * feat[c]
    a, bb, d = solve(A, b)
    a = a if a > 1e-9 else 1e-9
    ratio_da = d / a
    w0sq = math.sqrt(ratio_da) if ratio_da > 0 else 1.0
    c0 = 1.0 / math.sqrt(a)
    gamma0sq = bb / a + 2.0 * w0sq
    gamma0 = math.sqrt(gamma0sq) if gamma0sq > 0 else 0.1
    return c0, w0sq, gamma0


def main():
    data = sys.stdin.read().split()
    if not data:
        print("1.0"); return
    n = int(data[0])
    vals = data[2:]
    ws, amps = [], []
    for i in range(n):
        w = float(vals[3 * i]); xre = float(vals[3 * i + 1]); xim = float(vals[3 * i + 2])
        ws.append(w)
        amps.append(math.sqrt(xre * xre + xim * xim))
    wmax = max(ws)

    c0, w0sq0, gamma0_0 = crude_fit(ws, amps)

    best = (c0, w0sq0, gamma0_0)
    if _HAVE_SCIPY:
        lb = [1e-6, wmax * 1.02, 1e-6]
        ub = [10.0, wmax * 8.0, 5.0]
        rng = random.Random(777)
        best_cost = None
        starts = [[max(lb[0], min(ub[0], c0)), max(lb[1], min(ub[1], w0sq0)),
                   max(lb[2], min(ub[2], gamma0_0))]]
        for _ in range(11):
            starts.append([rng.uniform(lb[i], ub[i]) for i in range(3)])
        for x0 in starts:
            try:
                res = least_squares(residuals, x0, args=(ws, amps), method="trf",
                                     bounds=(lb, ub), max_nfev=1500)
            except Exception:
                continue
            if best_cost is None or res.cost < best_cost:
                best_cost = res.cost
                best = tuple(res.x)

    c, w0sq, gamma0 = best
    print("%.10g / sqrt((%.10g - w**2)**2 + (%.10g*w)**2)" % (c, w0sq, gamma0))


if __name__ == "__main__":
    main()
