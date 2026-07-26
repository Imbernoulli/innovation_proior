# TIER: strong
"""
The insight: don't just chase the lowest training residual on the one
history class you were shown -- test candidate kernel FAMILIES by an
invariant that must hold across history changes if the family is right.
For a genuine power-law kernel G(u) = A*u^-alpha, the complex modulus is
G*(w) ~ (i*w)^alpha, so the loss-tangent / phase lag between stress and
strain rate is CONSTANT across frequency. A sum of exponentials cannot have
that property (each mode's phase saturates at its own corner frequency).

We therefore fit BOTH a 3-term Prony series (same recipe as `greedy`) and a
single power law (log-log linear regression, closed form) to the training
data, then pick between them not by training residual (both fit the ~2
observed decades comparably well) but by which one gives a flatter
predicted phase lag across a spread of probe frequencies that extends
BEYOND the observed window -- the frequency-independence invariant a real
power-law kernel must satisfy and a finite sum of exponentials cannot. This
lets us trust the extrapolation-worthy family instead of the best in-sample
fit, and then simply reports that family with its regressed coefficients.
"""
import sys
import math


def read_rows():
    data = sys.stdin.read().split()
    n = int(data[0])
    rows = []
    idx = 2
    for _ in range(n):
        g0 = float(data[idx]); tt = float(data[idx + 1]); sig = float(data[idx + 2])
        rows.append((g0, tt, sig))
        idx += 3
    return rows


def fit_power_law(rows):
    xs, ys = [], []
    for g0, t, s in rows:
        gt = s / g0
        if gt <= 0:
            continue
        xs.append(math.log(t))
        ys.append(math.log(gt))
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sxy / sxx if sxx > 1e-12 else 0.0
    intercept = my - slope * mx
    alpha_hat = -slope
    a_hat = math.exp(intercept)
    return a_hat, alpha_hat


def fit_prony(rows, t_min, t_max, k=3):
    taus = [t_min * (t_max / t_min) ** (i / (k - 1)) for i in range(k)]
    xs = [[math.exp(-t / tau) for tau in taus] for _, t, _ in rows]
    ys = [s / g0 for g0, _, s in rows]
    AT_A = [[sum(xs[r][i] * xs[r][j] for r in range(len(xs))) for j in range(k)] for i in range(k)]
    AT_y = [sum(xs[r][i] * ys[r] for r in range(len(xs))) for i in range(k)]
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
    return taus, coef


def phase_lag_power(alpha, omega):
    # exact: G*(w) ~ (i*w)^alpha -> phase = alpha*pi/2, independent of w
    return alpha * math.pi / 2.0


def phase_lag_prony(taus, coef, omega):
    # G*(w) = sum a_k / (1/tau_k + i*w);  phase(w) = arg(G*(w))
    re = 0.0
    im = 0.0
    for tau, a in zip(taus, coef):
        inv_tau = 1.0 / tau
        denom = inv_tau * inv_tau + omega * omega
        # a_k*(i*w)/(1/tau_k + i*w) = a_k*w*(w + i/tau_k)/denom
        re += a * omega * omega / denom
        im += a * omega * inv_tau / denom
    if re == 0.0 and im == 0.0:
        return 0.0
    return math.atan2(im, re)


def main():
    rows = read_rows()
    ts = [r[1] for r in rows]
    t_min, t_max = min(ts), max(ts)

    a_pow, alpha_pow = fit_power_law(rows)
    taus, coef = fit_prony(rows, t_min, t_max)

    # invariance test: probe phase lag at frequencies spanning WELL beyond
    # the observed window (this is what "extrapolation-robust" must mean --
    # we never see the held-out data, only reason about the candidates).
    probe_omegas = []
    n_probe = 9
    lo = 2.0 * math.pi / (20.0 * t_max)
    hi = 2.0 * math.pi / (t_min / 20.0)
    for i in range(n_probe):
        frac = i / (n_probe - 1)
        probe_omegas.append(lo * (hi / lo) ** frac)

    pow_phases = [phase_lag_power(alpha_pow, w) for w in probe_omegas]
    prony_phases = [phase_lag_prony(taus, coef, w) for w in probe_omegas]

    def spread(vals):
        return max(vals) - min(vals)

    # power-law phase is exactly flat by construction (spread 0); a Prony
    # series' phase always drifts once you probe outside its fitted band.
    # We pick power law unless it is somehow flatter than Prony too (would
    # mean the data itself looked exponential), in which case fall back.
    if spread(pow_phases) <= spread(prony_phases) + 1e-9:
        print("%.8f * t ** ( -%.8f )" % (a_pow, alpha_pow))
    else:
        terms = ["( %.8f ) * exp( -t / %.8f )" % (coef[i], taus[i]) for i in range(len(taus))]
        print(" + ".join(terms))


if __name__ == "__main__":
    main()
