# TIER: strong
# The insight: efficacy is told to be a Hill/receptor-saturation curve
# E(d) = Emax*d^n/(EC50^n+d^n) -- the FORM is known, but Emax, EC50 (the
# saturation constant) and n are not. The training doses barely reach EC50, so
# a naive curve fit that ignores the ceiling (a quadratic, say) will keep
# extrapolating efficacy upward. Instead we NONLINEARLY fit the true Hill
# shape (recovering where it bends over -- the saturation constant), and
# separately fit toxicity as an accelerating power law Tbase + Tc*d^q (also a
# nonlinear fit, since no functional form was given for it -- only that it
# keeps climbing and accelerates). Locating EC50 is exactly what determines
# where the extrapolated efficacy curve stops rising fast enough to outrun the
# still-climbing toxicity curve, i.e. where the two curves' *trade-off*
# crosses over and the therapeutic window closes.
import sys
import numpy as np


def hill(x, Emax, EC50, n):
    EC50 = abs(EC50) + 1e-6
    n = abs(n) + 0.2
    xn = np.power(np.clip(x, 0, None), n)
    return Emax * xn / (EC50 ** n + xn)


def powlaw(x, Tbase, Tc, q):
    q = abs(q) + 0.2
    return Tbase + abs(Tc) * np.power(np.clip(x, 0, None), q)


def fit(func, d, y, p0, bounds):
    try:
        from scipy.optimize import curve_fit
        popt, _ = curve_fit(func, d, y, p0=p0, bounds=bounds, maxfev=20000)
        return [float(v) for v in popt]
    except Exception:
        return list(p0)


def main():
    data = sys.stdin.read().split()
    if len(data) < 3:
        print("EFFICACY 0")
        print("TOXICITY 0")
        return
    n = int(data[0])
    vals = data[3:]
    d = np.array([float(vals[3 * i]) for i in range(n)])
    e = np.array([float(vals[3 * i + 1]) for i in range(n)])
    tx = np.array([float(vals[3 * i + 2]) for i in range(n)])

    p0e = [max(e.max(), 1.0) * 1.5, float(np.median(d)) * 1.2, 1.5]
    popt_e = fit(hill, d, e, p0e, bounds=([1.0, 1.0, 0.5], [2000.0, 5000.0, 6.0]))
    Emax, EC50, nh = abs(popt_e[0]), abs(popt_e[1]) + 1e-6, abs(popt_e[2]) + 0.2

    p0t = [max(0.1, float(tx.min())), 0.01, 1.7]
    popt_t = fit(powlaw, d, tx, p0t, bounds=([0.0, 1e-8, 0.5], [200.0, 50.0, 6.0]))
    Tbase, Tc, q = abs(popt_t[0]), abs(popt_t[1]), abs(popt_t[2]) + 0.2

    # numeric coefficients kept as their own whitespace-separated tokens
    print("EFFICACY %.6f * d**%.6f / (%.6f**%.6f + d**%.6f)" % (Emax, nh, EC50, nh, nh))
    print("TOXICITY %.6f + %.6f * d**%.6f" % (Tbase, Tc, q))


if __name__ == "__main__":
    main()
