# TIER: strong
# The insight: stop fitting curves in T and instead SEARCH for the change of
# variables (Tc, phi) under which every field-sweep collapses onto one curve.
# For a TRIAL (Tc, phi, beta) the model
#     log m = log A + beta*log(u) - p*log(1+x^2),   u=|Tc-T|, x=h/u^phi, p=beta/(2phi)
# is LINEAR in log A once (Tc, beta, phi) are fixed, so log A has a closed
# form (mean residual) at every grid point.  Grid-search (Tc, beta, phi)
# coarse-to-fine to minimize the log-space sum of squares, then read off the
# exponents and amplitude directly -- this is the collapse: the training data
# never gets close to Tc, but the FUNCTIONAL FORM found this way is exact, so
# it extrapolates correctly to distances no training point reached and to the
# unseen T > Tc side, unlike any local polynomial patch.
import sys, math


def sse_for(rows, Tc, beta, phi):
    p = beta / (2.0 * phi)
    gs = []
    logms = []
    for T, h, m in rows:
        u = abs(Tc - T)
        if u < 1e-9:
            u = 1e-9
        x = h / (u ** phi)
        g = beta * math.log(u) - p * math.log(1.0 + x * x)
        gs.append(g)
        logms.append(math.log(m))
    n = len(rows)
    logA = sum(lm - g for lm, g in zip(logms, gs)) / n
    sse = sum((lm - g - logA) ** 2 for lm, g in zip(logms, gs))
    return sse, logA


def search(rows, tc_lo, tc_hi, beta_lo, beta_hi, phi_lo, phi_hi, steps):
    best = None
    for i in range(steps):
        Tc = tc_lo + (tc_hi - tc_lo) * i / (steps - 1)
        for j in range(steps):
            beta = beta_lo + (beta_hi - beta_lo) * j / (steps - 1)
            for k in range(steps):
                phi = phi_lo + (phi_hi - phi_lo) * k / (steps - 1)
                sse, logA = sse_for(rows, Tc, beta, phi)
                if best is None or sse < best[0]:
                    best = (sse, Tc, beta, phi, logA)
    return best


def main():
    data = sys.stdin.read().split()
    if not data:
        print("0.0"); return
    n = int(data[0])
    vals = data[2:]
    rows = []
    Tmax = -1e18
    for i in range(n):
        T = float(vals[3 * i]); h = float(vals[3 * i + 1]); m = float(vals[3 * i + 2])
        rows.append((T, h, m))
        if T > Tmax:
            Tmax = T

    # Every training row has T < Tc, so Tc must lie above the largest T seen.
    tc_lo, tc_hi = Tmax + 0.02, Tmax + 2.5
    beta_lo, beta_hi = 0.10, 0.90
    phi_lo, phi_hi = 0.50, 3.00

    best = search(rows, tc_lo, tc_hi, beta_lo, beta_hi, phi_lo, phi_hi, 9)
    for _ in range(2):
        _, Tc0, beta0, phi0, _ = best
        tc_w = (tc_hi - tc_lo) / 8.0 * 1.5
        beta_w = (beta_hi - beta_lo) / 8.0 * 1.5
        phi_w = (phi_hi - phi_lo) / 8.0 * 1.5
        tc_lo2, tc_hi2 = max(Tmax + 0.001, Tc0 - tc_w), Tc0 + tc_w
        beta_lo2, beta_hi2 = max(0.02, beta0 - beta_w), min(1.5, beta0 + beta_w)
        phi_lo2, phi_hi2 = max(0.1, phi0 - phi_w), min(5.0, phi0 + phi_w)
        best = search(rows, tc_lo2, tc_hi2, beta_lo2, beta_hi2, phi_lo2, phi_hi2, 7)

    sse, Tc, beta, phi, logA = best
    A = math.exp(logA)
    p = beta / (2.0 * phi)
    print("%.10g * absv(T-(%.10g))**(%.10g) * (1.0+(h/absv(T-(%.10g))**(%.10g))**2)**(-(%.10g))"
          % (A, Tc, beta, Tc, phi, p))


if __name__ == "__main__":
    main()
