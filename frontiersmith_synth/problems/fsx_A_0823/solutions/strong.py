# TIER: strong
# The insight: stop fitting the amplitude curve and instead COMMIT to the
# physical model CLASS -- a two-coupled-oscillator transfer function with
# FIVE real constants (Om1, Om2, g1, g2, kc) -- and fit ALL of it directly
# against the phase-resolved (Xre, Xim) lock-in data, not just its modulus.
# A single Lorentzian (greedy's three numbers) cannot represent a second
# pole pair no matter how it is tuned; the two-mode form CAN, and because it
# is the correct functional class, even an imperfect fit to noisy tail-only
# data reproduces both resonance peaks, the avoided-crossing splitting
# between them, and the correct high-frequency 1/w^4 roll-off once
# extrapolated -- none of which a wrong model class can ever recover, no
# matter how well it is optimized.
#
# The five-parameter fit is genuinely non-convex (multiple local optima look
# similarly good on tail-only data), so this uses bounded multi-start
# nonlinear least squares (deterministic seed) and keeps the lowest-cost
# solution -- a reformulation + search strategy, not "greedy plus more
# iterations": greedy never had a slot for a second natural frequency to
# begin with.
import sys, math, random

try:
    from scipy.optimize import least_squares
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def response(w, Om1, Om2, g1, g2, kc):
    a1 = Om1 * Om1 - w * w
    a2 = Om2 * Om2 - w * w
    b1 = g1 * w
    b2 = g2 * w
    Dre = a1 * a2 - b1 * b2 - kc * kc
    Dim = a1 * b2 + a2 * b1
    Dmag2 = Dre * Dre + Dim * Dim
    if Dmag2 < 1e-300:
        Dmag2 = 1e-300
    Xre = (a2 * Dre + b2 * Dim) / Dmag2
    Xim = (b2 * Dre - a2 * Dim) / Dmag2
    return Xre, Xim


def residuals(params, ws, xres, xims):
    Om1, Om2, g1, g2, kc = params
    out = []
    for w, xre, xim in zip(ws, xres, xims):
        pr, pi = response(w, Om1, Om2, g1, g2, kc)
        out.append(pr - xre)
        out.append(pi - xim)
    return out


def crude_single_mode(ws, amps):
    """Closed-form single-pole estimate, reused as one of the multi-start
    seeds (a two-mode fit initialized near the best one-mode compromise is a
    sensible starting point, not the final answer)."""
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
    gamma0sq = bb / a + 2.0 * w0sq
    gamma0 = math.sqrt(gamma0sq) if gamma0sq > 0 else 0.2
    return w0sq, gamma0


def main():
    data = sys.stdin.read().split()
    if not data:
        print("1.0"); return
    n = int(data[0])
    vals = data[2:]
    ws, xres, xims, amps = [], [], [], []
    for i in range(n):
        w = float(vals[3 * i]); xre = float(vals[3 * i + 1]); xim = float(vals[3 * i + 2])
        ws.append(w); xres.append(xre); xims.append(xim)
        amps.append(math.sqrt(xre * xre + xim * xim))
    wmax = max(ws)

    w0sq_guess, gamma_guess = crude_single_mode(ws, amps)
    w0_guess = math.sqrt(w0sq_guess) if w0sq_guess > 0 else wmax * 2.0

    lb = [wmax * 1.05, wmax * 1.10, 0.01, 0.01, 0.0]
    ub = [wmax * 6.0, wmax * 9.0, 3.0, 3.0, 15.0]

    def clamp(v, i):
        return max(lb[i], min(ub[i], v))

    best = None
    if _HAVE_SCIPY:
        rng = random.Random(12345)
        starts = [[clamp(w0_guess, 0), clamp(w0_guess * 1.4, 1),
                   clamp(gamma_guess, 2), clamp(gamma_guess, 3), clamp(1.0, 4)]]
        for _ in range(17):
            starts.append([rng.uniform(lb[i], ub[i]) for i in range(5)])
        for x0 in starts:
            try:
                res = least_squares(residuals, x0, args=(ws, xres, xims), method="trf",
                                     bounds=(lb, ub), max_nfev=1500)
            except Exception:
                continue
            if best is None or res.cost < best[0]:
                best = (res.cost, tuple(res.x))

    if best is None:
        # scipy unavailable: fall back to the single-mode compromise, widened
        # slightly into a nominal second mode so the expression stays a
        # genuine two-mode form (still far better than pure greedy on the
        # resonance region for well-separated cases).
        Om1f = clamp(w0_guess, 0)
        Om2f = clamp(w0_guess * 1.4, 1)
        g1f = clamp(gamma_guess, 2)
        g2f = clamp(gamma_guess, 3)
        kcf = clamp(0.5, 4)
    else:
        Om1f, Om2f, g1f, g2f, kcf = best[1]

    Om1sq, Om2sq = Om1f * Om1f, Om2f * Om2f
    expr = (
        "sqrt((%.10g - w**2)**2 + (%.10g*w)**2)"
        " / sqrt((((%.10g - w**2)*(%.10g - w**2) - (%.10g)*(%.10g)*w**2 - (%.10g)**2))**2"
        " + ((%.10g)*w*(%.10g - w**2) + (%.10g)*w*(%.10g - w**2))**2)"
    ) % (Om2sq, g2f, Om1sq, Om2sq, g1f, g2f, kcf, g1f, Om2sq, g2f, Om1sq)
    print(expr)


if __name__ == "__main__":
    main()
